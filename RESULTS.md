# Results log

One row per run. `python -m lifemate.evaluate runs/<run>` (and the molab notebook) prints the row ready to paste.
Train scores are local (`lifemate.scoring`); public_test scores come from the leaderboard, so fill in "score" by hand.
Timing is per sample on the machine that ran it; the scoring A30 is slower than molab's RTX Pro 6000.

| date | commit | split | n | agent | asr | score | F1 no hash | act | ask | empty | mean s | max s | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
