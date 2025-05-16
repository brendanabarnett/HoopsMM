```
data/
├── embeddings_sample.csv             # Sentence embeddings of each game's summary (sample only)
├── fused.csv                         # Tabular features + summary embeddings for each team’s past 5 games (with binary outcome)
├── odds_h2h.csv                      # Pre-game moneyline betting odds for both teams
├── ranking_all.csv                   # Season-level stats, NCAA ranks, and March Madness outcomes (all teams)
├── ranking_top.csv                   # Subset of `ranking_all.csv` including only ranked/top-performing teams
├── tabular_core.csv                  # Core tabular features from box scores: rankings, recent form, match context
├── text_concise_summary_metrics.csv  # GPT-generated summaries highlighting team-level game stats (1–2 sentences)
└── text_play_by_play_team.csv        # Team-level play-by-play logs with second-level timestamps for all key events
```
