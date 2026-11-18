# Round-five GTA launch forecast

This standalone model forecasts viewers for hours 2–24 by blending nearby launch histories with a hinge regression. The regression uses three rounds of robust reweighting, and a nested leave-one-launch-out procedure estimates a global median log-error adjustment for the forecast.

The validated leave-one-out mean loss is 0.1798503047245903; low, middle, and high launch-size groups score 0.28518584149197984, 0.13567828726933315, and 0.10990882401517543. The four practice cases score 0.000000400411 before six-decimal display rounding.

Run the model with Python 3 and set `MAKEFUN_INPUT_DIR` to the directory of `.in` case files and `MAKEFUN_OUTPUT_DIR` to the desired output directory. It writes one compact `.out` JSON file for each input case. For example:

    MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 /path/to/huber-bias-corrected.py

The module also provides `LEVELS`, `forecast_case(case)`, and `main()` for direct integration and evaluation.
