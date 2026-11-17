# GTA VI Twitch Forecast — make.fun bounty entry

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/auroranightseeu/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/auroranightseeu/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)


Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers, hour by
hour, for the make.fun bounty
["Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
(`gta6-twitch-ccu-forecast-01`).

- Prize: **$250 total, top 20 paid** (1st $50, 2nd $35, 3rd $25, 4th–10th $10,
  11th–20th $7), in USDC on Solana. A prize needs a score better than the best
  house entry.
- Deadline: **Thu 19 Nov 2026, 01:00 UTC (08:00 WIB)**. Max 10 submissions/day,
  20 total; the last submission before the deadline is the final entry.

## What this is

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads one case from `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`): 23 rows, hours 2–24,
five quantiles each (5%, 25%, 50%, 75%, 95%).

Key finding from the kit data: **how much viewership grows after hour 1
depends strongly on how big hour 1 already is** (correlation −0.93 between
log hour-1 viewers and the log growth to hour 19). Small hour-1 launches
explode later; huge hour-1 launches stay flat. The flat house baseline
ignores this.

The model, fitted at runtime from the case's own `comparables`, blends
kNN (k=7) empirical quantiles with a hinge regression on log hour-1 viewers
(knot at ln 60,000), 50/50, falling back to flat persistence when too few
comparables exist.

## Verified locally (2026-11-17)

- Leave-one-out pinball loss on 37 comparables: **0.1883**, vs flat baseline
  **0.4648** — 59% better.
- 4 practice cases: 23 rows each, 5 quantiles, 0.19 s total (limit 8 s/case),
  byte-identical on re-run.
- Real practice score and the house AI baseline can only be measured by
  make.fun's grader — beating the flat baseline locally is necessary, not
  a guarantee of a prize.

## Run locally

```bash
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```
