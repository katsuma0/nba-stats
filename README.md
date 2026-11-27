# nba-stats

Who actually won the trade? This repo scores NBA trades since 2000 by what the incoming players gave their new team.

```bash
python3 analysis/trade_impact.py
```

- `data/raw/`: Basketball-Reference season tables (via sumitrodatta/bball-reference-datasets)
- `data/trades_curated.csv`: 57 landmark trades, every side listed
- `output/curated_trade_sides.csv`: WS, VORP, margin and win % change per side
- `output/midseason_moves.csv`: every mid-season team change, BPM before vs after, tenure value
- `output/trade_report.html`: the visual report
