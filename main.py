"""Forecast launch-day viewers by blending two views of past launches.

For each hour, the forecast averages (in log space) a k-nearest-neighbours
forecast — empirical quantiles of the log growth of the 7 comparables whose
hour-1 viewers are closest in log space — with a hinge-regression forecast
(log growth regressed on log hour-1 viewers, slope allowed to change at
ln(60000), normal quantiles from the residual spread). A flat persistence
baseline is the fallback when too few neighbours exist.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

K_NEIGHBOURS = 7
HINGE_KNOT = 11.0
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def round2(x):
    rounded = math.floor(x * 100.0 + 0.5) / 100.0
    return int(rounded) if rounded.is_integer() else rounded


def empirical_quantile(values, level):
    index = level * (len(values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def _solve_3x3(a, b):
    if not all(math.isfinite(value) for row in a for value in row):
        return None
    if not all(math.isfinite(value) for value in b):
        return None
    matrix = [list(a[row]) + [b[row]] for row in range(3)]

    for column in range(3):
        pivot = max(
            range(column, 3),
            key=lambda row: abs(matrix[row][column]),
        )
        if abs(matrix[pivot][column]) < 1e-12:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]

        for row in range(column + 1, 3):
            factor = matrix[row][column] / matrix[column][column]
            matrix[row][column] = 0.0
            for index in range(column + 1, 4):
                matrix[row][index] -= factor * matrix[column][index]

    solution = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        if abs(matrix[row][row]) < 1e-12:
            return None
        solution[row] = (
            matrix[row][3]
            - sum(matrix[row][column] * solution[column]
                  for column in range(row + 1, 3))
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
        ]
        hinge_comparables.append((features, ratios))

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    neighbours = candidates[:K_NEIGHBOURS]
    hours = []
    minimum_log = math.log(0.01)
    maximum_log = math.log(1e9)
    case_features = [
        1.0,
        min(logp1, HINGE_KNOT),
        max(0.0, logp1 - HINGE_KNOT),
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
                xtx = [[0.0] * 3 for _ in range(3)]
                xty = [0.0] * 3
                for features, target in training:
                    for row in range(3):
                        xty[row] += features[row] * target
                        for column in range(3):
                            xtx[row][column] += features[row] * features[column]
                for index in range(3):
                    xtx[index][index] += 1e-8

                beta = _solve_3x3(xtx, xty)
                if beta is not None and all(
                    math.isfinite(value) for value in beta
                ):
                    residuals = [
                        target - sum(
                            beta[index] * features[index]
                            for index in range(3)
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
                            for index in range(3)
                        )
                        hinge_log_quantiles = [
                            logp1 + pred + z_score * sigma
                            for z_score in Z_SCORES
                        ]
                        log_quantiles = [
                            0.5 * (knn_log + hinge_log)
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
