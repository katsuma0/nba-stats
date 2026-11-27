"""NBA trade impact analysis, 2000 to present.

Two views:
  1. Curated trades (data/trades_curated.csv): both sides scored by the Win Shares
     and VORP each franchise actually got from the players it received, for as long
     as those players stayed. Team win% before and after is attached too.
  2. Every mid-season move since 2000, detected from multi-team stat lines: how
     each player produced before vs after, and what he gave his new franchise.

Source: Basketball-Reference season tables via github.com/sumitrodatta/bball-reference-datasets.
Run from the repo root:  python analysis/trade_impact.py
"""
import json
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

FIRST, LAST_FULL = 2000, 2026  # 2027 is in progress; counted in tenure, never as a "season after"

# Relocated or renamed teams collapse into one franchise code.
FRANCHISE = {"SEA": "OKC", "NJN": "BRK", "VAN": "MEM", "NOH": "NOP", "NOK": "NOP",
             "CHH": "NOP", "CHA": "CHO", "WSB": "WAS"}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return s.lower().replace(".", "").replace("'", "").strip()


def load():
    adv = pd.read_csv(RAW / "advanced.csv")
    adv = adv[(adv.lg == "NBA") & (adv.season >= FIRST - 1)].copy()
    adv["fr"] = adv.team.replace(FRANCHISE)
    adv["key"] = adv.player.map(norm)
    adv["stint"] = adv.groupby(["season", "player_id"]).cumcount()
    teams = pd.read_csv(RAW / "team_summaries.csv")
    teams = teams[(teams.lg == "NBA") & teams.abbreviation.notna()].copy()
    teams["fr"] = teams.abbreviation.replace(FRANCHISE)
    teams["win_pct"] = teams.w / (teams.w + teams.l)
    return adv, teams


def tenure(rows: pd.DataFrame, fr: str, start: int) -> pd.DataFrame:
    """Stat lines a player gave franchise `fr` from `start` until he left (1-season gap allowed)."""
    mine = rows[(rows.fr == fr) & (rows.season >= start)].sort_values(["season", "stint"])
    if mine.empty or mine.season.iloc[0] > start + 4:
        return mine.iloc[0:0]
    keep, prev = [], None
    for idx, r in mine.iterrows():
        if prev is not None and r.season - prev > 2:
            break
        keep.append(idx)
        prev = r.season
    return mine.loc[keep]


def team_record(teams, fr, season):
    r = teams[(teams.fr == fr) & (teams.season == season)]
    return (None, None) if r.empty else (round(float(r.win_pct.iloc[0]), 3), float(r.n_rtg.iloc[0]))


def curated(adv, teams):
    trades = pd.read_csv(ROOT / "data" / "trades_curated.csv")
    by_player = {k: g for k, g in adv[~adv.team.str.endswith("TM")].groupby("key")}
    sides, missing = [], []
    for _, t in trades.iterrows():
        before_s = t.season - 1
        after_s = t.season + 1 if t.midseason else t.season
        players = []
        for name in t.received.split(";"):
            rows = by_player.get(norm(name))
            ten = tenure(rows, t.team, t.season) if rows is not None else None
            if ten is None or (ten.empty and rows is None):
                missing.append(name)
                continue
            # Mid-season: only count the stint after the trade in the trade season.
            if t.midseason and not ten.empty:
                first = ten[ten.season == t.season]
                if len(first) > 1:
                    ten = ten.drop(first.index[:-1])
            players.append({
                "player": name, "seasons": int(ten.season.nunique()),
                "games": int(ten.g.sum()), "ws": round(float(ten.ws.sum()), 1),
                "vorp": round(float(ten.vorp.sum()), 1),
                "bpm": round(float((ten.bpm * ten.mp).sum() / ten.mp.sum()), 1) if ten.mp.sum() else None,
            })
        wb, nb = team_record(teams, t.team, before_s)
        wa, na = team_record(teams, t.team, after_s) if after_s <= LAST_FULL else (None, None)
        sides.append({
            "trade_id": t.trade_id, "season": t.season, "label": t.label, "team": t.team,
            "midseason": bool(t.midseason),
            "ws": round(sum(p["ws"] for p in players), 1),
            "vorp": round(sum(p["vorp"] for p in players), 1),
            "win_pct_before": wb, "win_pct_after": wa,
            "win_pct_change": round(wa - wb, 3) if wa is not None and wb is not None else None,
            "nrtg_change": round(na - nb, 1) if na is not None and nb is not None else None,
            "players": players,
        })
    if missing:
        raise SystemExit(f"unmatched players: {missing}")
    df = pd.DataFrame(sides)
    # Margin: what this side got minus the best haul any other side got.
    df["margin_ws"] = df.apply(
        lambda r: round(r.ws - df[(df.trade_id == r.trade_id) & (df.team != r.team)].ws.max(), 1), axis=1)
    return df


def midseason_moves(adv):
    """Every player with 2+ team stints in a season, scored on the move into each new team."""
    rows = adv[(~adv.team.str.endswith("TM")) & adv.season.between(FIRST, LAST_FULL + 1)]
    multi = rows.groupby(["season", "player_id"]).filter(lambda g: len(g) > 1)
    by_player = {k: g for k, g in adv[~adv.team.str.endswith("TM")].groupby("player_id")}
    out = []
    for (season, pid), g in multi.groupby(["season", "player_id"]):
        g = g.sort_values("stint")
        for i in range(1, len(g)):
            a, b = g.iloc[i - 1], g.iloc[i]
            if a.fr == b.fr:
                continue
            ten = tenure(by_player[pid], b.fr, season)
            ten = ten[~((ten.season == season) & (ten.stint < b.stint))]
            out.append({
                "season": int(season), "player": b.player, "from": a.fr, "to": b.fr,
                "mp_before": int(a.mp), "mp_after": int(b.mp),
                "bpm_before": a.bpm, "bpm_after": b.bpm, "bpm_change": round(b.bpm - a.bpm, 1),
                "ws48_before": a.ws_48, "ws48_after": b.ws_48,
                "tenure_seasons": int(ten.season.nunique()),
                "tenure_ws": round(float(ten.ws.sum()), 1),
                "tenure_vorp": round(float(ten.vorp.sum()), 1),
            })
    return pd.DataFrame(out)


def main():
    adv, teams = load()
    cur = curated(adv, teams)
    mid = midseason_moves(adv)

    cur.drop(columns="players").to_csv(OUT / "curated_trade_sides.csv", index=False)
    mid.to_csv(OUT / "midseason_moves.csv", index=False)

    qualified = mid[(mid.mp_before >= 400) & (mid.mp_after >= 400)]
    report = {
        "generated_through_season": LAST_FULL,
        "trades": cur.to_dict("records"),
        "midseason_count": int(len(mid)),
        "top_tenure": mid.nlargest(25, "tenure_ws").to_dict("records"),
        "top_jump": qualified.nlargest(20, "bpm_change").to_dict("records"),
        "top_drop": qualified.nsmallest(20, "bpm_change").to_dict("records"),
        "moves_by_season": mid.groupby("season").size().to_dict(),
    }
    (OUT / "report_data.json").write_text(json.dumps(report, default=str, indent=1))
    page = (ROOT / "analysis" / "report_template.html").read_text()
    blob = json.dumps({k: v for k, v in report.items() if k != "moves_by_season"},
                      default=str, separators=(",", ":")).replace("</", "<\\/")
    (OUT / "trade_report.html").write_text(page.replace("/*DATA*/null", blob))

    best = cur.sort_values("margin_ws", ascending=False)
    print(f"{len(cur.trade_id.unique())} curated trades, {len(mid)} mid-season moves")
    print("\nBest franchise hauls (WS margin):")
    print(best.head(12)[["season", "label", "team", "ws", "margin_ws", "win_pct_change"]].to_string(index=False))
    print("\nWorst franchise hauls:")
    print(best.tail(12)[["season", "label", "team", "ws", "margin_ws", "win_pct_change"]].to_string(index=False))
    print("\nMost WS for new team after a mid-season move:")
    print(mid.nlargest(12, "tenure_ws")[["season", "player", "from", "to", "tenure_seasons", "tenure_ws"]].to_string(index=False))


if __name__ == "__main__":
    main()
