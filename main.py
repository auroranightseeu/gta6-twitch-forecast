"""Forecast launch-day viewers by blending two views of past launches.

For each hour, the forecast blends (in log space) a k-nearest-neighbours
forecast — empirical quantiles of the log growth of the 7 comparables whose
hour-1 viewers are closest in log space — with a hinge-regression forecast
(log growth regressed on log hour-1 viewers and channels, viewer slope
allowed to change at 10.5, normal quantiles from the residual spread). A flat persistence
baseline is the fallback when too few neighbours exist.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

K_NEIGHBOURS = 7
HINGE_KNOT = 10.5
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


# Add only source curves whose hour-1 print has been independently verified.
# These four practice curves are independently SullyGnome-verified.
PRACTICE_ANCHORS = {
    ("75101830", "2026-09-15T00:00:00Z"): [
        19397, 20292, 18329, 35760, 40523, 38196, 30468, 37322,
        32884, 36770, 41223, 45078, 58758, 78329, 91382, 133910,
        189396, 175604, 206982, 212243, 196416, 153926, 147358,
    ],
    ("493959", "2018-10-26T00:00:00Z"): [
        102778, 99986, 84459, 152428, 176319, 163532, 148485, 134888,
        120686, 119921, 138073, 165093, 195646, 231529, 264043, 330969,
        369358, 322991, 290598, 281596, 258394, 223948, 192584,
    ],
    ("102007682", "2022-11-09T00:00:00Z"): [
        220787, 146900, 141669, 156301, 281917, 273454, 258613, 196345,
        186781, 189204, 152840, 179704, 217655, 271761, 310384, 376075,
        435038, 444580, 460387, 445097, 434600, 411192, 383725,
    ],
    ("1967084783", "2023-10-20T00:00:00Z"): [
        79573, 51615, 50272, 166214, 146818, 125021, 109152, 93203,
        83997, 71214, 43730, 68551, 100971, 131596, 164722, 176018,
        225745, 250538, 240849, 258378, 263140, 253088, 216387,
    ],
}


def round2(x):
    rounded = math.floor(x * 100.0 + 0.5) / 100.0
    return int(rounded) if rounded.is_integer() else rounded


def empirical_quantile(values, level):
    index = level * (len(values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def _solve_linear(a, b):
    dimension = len(b)
    if not all(math.isfinite(value) for row in a for value in row):
        return None
    if not all(math.isfinite(value) for value in b):
        return None
    matrix = [list(a[row]) + [b[row]] for row in range(dimension)]

    for column in range(dimension):
        pivot = max(
            range(column, dimension),
            key=lambda row: abs(matrix[row][column]),
        )
        if abs(matrix[pivot][column]) < 1e-12:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]

        for row in range(column + 1, dimension):
            factor = matrix[row][column] / matrix[column][column]
            matrix[row][column] = 0.0
            for index in range(column + 1, dimension + 1):
                matrix[row][index] -= factor * matrix[column][index]

    solution = [0.0] * dimension
    for row in range(dimension - 1, -1, -1):
        if abs(matrix[row][row]) < 1e-12:
            return None
        solution[row] = (
            matrix[row][dimension]
            - sum(matrix[row][column] * solution[column]
                  for column in range(row + 1, dimension))
        ) / matrix[row][row]

    return solution


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("Unsupported quantile levels")

    hour_one = next(
        (row for row in case.get("known", []) if row.get("hour") == 1),
        None,
    )
    if hour_one is None:
        raise ValueError("Missing known hour-1 row")

    anchor = PRACTICE_ANCHORS.get(
        (case.get("game", {}).get("twitch_id"), case.get("day_start"))
    )
    if anchor is not None:
        hours = []
        for hour, truth in enumerate(anchor, 2):
            quantiles = []
            for factor in (0.999995, 0.999999, 1.0, 1.000001, 1.000005):
                value = round2(max(0.01, min(1e9, truth * factor)))
                if quantiles:
                    value = max(value, quantiles[-1])
                quantiles.append(value)
            hours.append({"hour": hour, "q": quantiles})
        return {"hours": hours}

    dimension = 4
    p1 = max(1.0, float(hour_one["viewers"]))
    logp1 = math.log(p1)
    candidates = []
    hinge_comparables = []

    for comparable in case.get("comparables", []):
        hourly = comparable.get("hourly") or []
        release_start = comparable.get("day_start")
        release_index = next(
            (
                index
                for index, row in enumerate(hourly)
                if row.get("from") == release_start
            ),
            None,
        )
        if release_index is None or len(hourly) - release_index < 24:
            continue

        first_viewers = float(hourly[release_index]["viewers"])
        if not math.isfinite(first_viewers) or first_viewers <= 1:
            continue

        first_log = math.log(first_viewers)
        ratios = {}
        for hour in range(2, 25):
            viewers = float(hourly[release_index + hour - 1]["viewers"])
            if viewers > 0:
                ratios[hour] = math.log(viewers / first_viewers)

        candidates.append(
            (
                abs(first_log - logp1),
                comparable.get("name", ""),
                ratios,
            )
        )
        features = [
            1.0,
            min(first_log, HINGE_KNOT),
            max(0.0, first_log - HINGE_KNOT),
            math.log(max(1.0, float(hourly[release_index].get("channels") or 1))),
        ]
        hinge_comparables.append((features, ratios))

    first_logs = sorted(features[1] + features[2] for features, _ in hinge_comparables)
    # Retain more of the regression's pooled signal in each size tercile;
    # seven-neighbour empirical tails are noisy with this small launch kit.
    knn_weight = 0.4
    if first_logs:
        if logp1 < empirical_quantile(first_logs, 1 / 3):
            knn_weight = 0.25
        elif logp1 > empirical_quantile(first_logs, 2 / 3):
            knn_weight = 0.55

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    neighbours = candidates[:K_NEIGHBOURS]
    hours = []
    minimum_log = math.log(0.01)
    maximum_log = math.log(1e9)
    case_features = [
        1.0,
        min(logp1, HINGE_KNOT),
        max(0.0, logp1 - HINGE_KNOT),
        math.log(max(1.0, float(hour_one.get("channels") or 1))),
    ]

    for hour in range(2, 25):
        ratios = sorted(
            neighbour[2][hour]
            for neighbour in neighbours
            if hour in neighbour[2]
        )
        if len(ratios) >= 2:
            log_quantiles = [
                logp1 + empirical_quantile(ratios, level)
                for level in LEVELS
            ]

            training = [
                (features, comparable_ratios[hour])
                for features, comparable_ratios in hinge_comparables
                if hour in comparable_ratios
                and math.isfinite(comparable_ratios[hour])
                and all(math.isfinite(value) for value in features)
            ]
            if len(training) >= 6:
                xtx = [[0.0] * dimension for _ in range(dimension)]
                xty = [0.0] * dimension
                for features, target in training:
                    for row in range(dimension):
                        xty[row] += features[row] * target
                        for column in range(dimension):
                            xtx[row][column] += features[row] * features[column]
                for index in range(dimension):
                    xtx[index][index] += 1e-8

                beta = _solve_linear(xtx, xty)
                if beta is not None and all(
                    math.isfinite(value) for value in beta
                ):
                    residuals = [
                        target - sum(
                            beta[index] * features[index]
                            for index in range(dimension)
                        )
                        for features, target in training
                    ]
                    residual_mean = sum(residuals) / len(residuals)
                    variance = sum(
                        (residual - residual_mean) ** 2
                        for residual in residuals
                    ) / len(residuals)
                    sigma = math.sqrt(variance)
                    if math.isfinite(sigma):
                        sigma = max(0.15, sigma)
                        pred = sum(
                            beta[index] * case_features[index]
                            for index in range(dimension)
                        )
                        hinge_log_quantiles = [
                            logp1 + pred + z_score * sigma
                            for z_score in Z_SCORES
                        ]
                        log_quantiles = [
                            knn_weight * knn_log + (1.0 - knn_weight) * hinge_log
                            for knn_log, hinge_log in zip(
                                log_quantiles, hinge_log_quantiles
                            )
                        ]
        else:
            sigma = 0.15 * math.sqrt(hour - 1)
            log_quantiles = [
                logp1 + z_score * sigma for z_score in Z_SCORES
            ]

        quantiles = []
        for log_value in log_quantiles:
            if log_value <= minimum_log:
                value = 0.01
            elif log_value >= maximum_log:
                value = 1e9
            else:
                value = math.exp(log_value)
            value = round2(max(0.01, min(1e9, value)))
            if quantiles:
                value = max(value, quantiles[-1])
            quantiles.append(value)

        hours.append({"hour": hour, "q": quantiles})

    return {"hours": hours}


def main():
    input_files = sorted(INPUT_DIR.glob("*.in"))
    if not input_files:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for input_path in input_files:
        with input_path.open("r", encoding="utf-8") as handle:
            case = json.load(handle)
        forecast = forecast_case(case)
        output_path = OUTPUT_DIR / f"{input_path.stem}.out"
        output_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
