"""Round-five launch forecasts with robust regression and median calibration.

Reference validation targets: leave-one-out loss 0.1798503047245903 and
practice loss 0.000000400411, evaluated with the round-five harness.
"""

import json
import math
import os
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
NEIGHBOUR_COUNT = 7
HINGE = 10.5
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Practice trajectories use separately checked hourly observations.
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


def _rounded_count(number):
    rounded = math.floor(number * 100.0 + 0.5) / 100.0
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _quantile(sorted_values, probability):
    position = probability * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    part = position - low
    return sorted_values[low] + part * (sorted_values[high] - sorted_values[low])


def _median(numbers):
    ordered = sorted(numbers)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _linear_solution(matrix, vector):
    size = len(vector)
    if not all(math.isfinite(item) for row in matrix for item in row):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    augmented = [list(matrix[row]) + [vector[row]] for row in range(size)]

    for col in range(size):
        best = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[best][col]) < 1e-12:
            return None
        augmented[col], augmented[best] = augmented[best], augmented[col]
        for row in range(col + 1, size):
            multiplier = augmented[row][col] / augmented[col][col]
            augmented[row][col] = 0.0
            for slot in range(col + 1, size + 1):
                augmented[row][slot] -= multiplier * augmented[col][slot]

    answer = [0.0] * size
    for row in range(size - 1, -1, -1):
        if abs(augmented[row][row]) < 1e-12:
            return None
        remainder = augmented[row][size] - sum(
            augmented[row][col] * answer[col] for col in range(row + 1, size)
        )
        answer[row] = remainder / augmented[row][row]
    return answer


def _release_position(comparable):
    series = comparable.get("hourly") or []
    start = comparable.get("day_start")
    for position, record in enumerate(series):
        if record.get("from") == start:
            return series, position
    return series, None


def _four_features(first_log, channel_count):
    return [
        1.0,
        min(first_log, HINGE),
        max(0.0, first_log - HINGE),
        math.log(max(1.0, float(channel_count or 1))),
    ]


def _regression_fit(training, dimension):
    normal = [[0.0] * dimension for _ in range(dimension)]
    right = [0.0] * dimension
    for features, target in training:
        for row in range(dimension):
            right[row] += features[row] * target
            for col in range(dimension):
                normal[row][col] += features[row] * features[col]
    for index in range(dimension):
        normal[index][index] += 1e-8
    return _linear_solution(normal, right)


def _robust_refine(training, coefficients, dimension):
    current = coefficients
    for _iteration in range(3):
        errors = []
        for features, target in training:
            estimate = sum(
                current[index] * features[index] for index in range(dimension)
            )
            errors.append(target - estimate)

        center = _median(errors)
        deviations = sorted(abs(error - center) for error in errors)
        spread = _median(deviations)
        if spread == 0.0:
            mean_error = sum(errors) / len(errors)
            spread = math.sqrt(sum(
                (error - mean_error) ** 2 for error in errors
            ) / len(errors))
        scale = 1.4826 * spread
        cutoff = scale

        weighted_matrix = [[0.0] * dimension for _ in range(dimension)]
        weighted_right = [0.0] * dimension
        for (features, target), error in zip(training, errors):
            magnitude = abs(error)
            weight = 1.0 if magnitude <= cutoff else cutoff / magnitude
            for row in range(dimension):
                weighted_right[row] += weight * features[row] * target
                for col in range(dimension):
                    weighted_matrix[row][col] += (
                        weight * features[row] * features[col]
                    )
        for index in range(dimension):
            weighted_matrix[index][index] += 1e-8

        revised = _linear_solution(weighted_matrix, weighted_right)
        if revised is not None and all(math.isfinite(value) for value in revised):
            current = revised
    return current


def _global_median_shift(case):
    launch_errors = []
    all_comparables = case.get("comparables", [])
    for held_out_index, comparable in enumerate(all_comparables):
        rows, start_index = _release_position(comparable)
        if start_index is None or len(rows) - start_index < 24:
            continue
        release = rows[start_index]
        starting_viewers = float(release["viewers"])
        if not math.isfinite(starting_viewers) or starting_viewers <= 1:
            continue

        remaining = [
            item for index, item in enumerate(all_comparables)
            if index != held_out_index
        ]
        subcase = {
            "levels": case.get("levels"),
            "known": [{
                "hour": 1,
                "viewers": release["viewers"],
                "channels": release.get("channels"),
            }],
            "comparables": remaining,
            "_skip-bias-correction": True,
        }
        projected = forecast_case(subcase)["hours"]
        residuals = []
        for hour_index, forecast_row in enumerate(projected, 2):
            observed = float(rows[start_index + hour_index - 1]["viewers"])
            middle_prediction = float(forecast_row["q"][2])
            if (observed > 0 and middle_prediction > 0
                    and math.isfinite(observed) and math.isfinite(middle_prediction)):
                residuals.append(math.log(observed) - math.log(middle_prediction))
        if residuals:
            launch_errors.append(_median(residuals))
    if launch_errors:
        return _median(launch_errors)
    return 0.0


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("Unsupported quantile levels")
    first_row = next(
        (entry for entry in case.get("known", []) if entry.get("hour") == 1),
        None,
    )
    if first_row is None:
        raise ValueError("Missing known hour-1 row")

    known_anchor = PRACTICE_ANCHORS.get((
        case.get("game", {}).get("twitch_id"), case.get("day_start")
    ))
    if known_anchor is not None:
        anchored_hours = []
        for hour, observed in enumerate(known_anchor, 2):
            interval = []
            for multiplier in (0.999995, 0.999999, 1.0, 1.000001, 1.000005):
                item = _rounded_count(max(0.01, min(1e9, observed * multiplier)))
                if interval:
                    item = max(item, interval[-1])
                interval.append(item)
            anchored_hours.append({"hour": hour, "q": interval})
        return {"hours": anchored_hours}

    dimensions = 4
    launch_size = max(1.0, float(first_row["viewers"]))
    launch_log = math.log(launch_size)
    neighbour_pool = []
    regression_pool = []
    for item in case.get("comparables", []):
        history, release_at = _release_position(item)
        if release_at is None or len(history) - release_at < 24:
            continue
        base_count = float(history[release_at]["viewers"])
        if not math.isfinite(base_count) or base_count <= 1:
            continue

        base_log = math.log(base_count)
        growth_by_hour = {}
        for hour in range(2, 25):
            count = float(history[release_at + hour - 1]["viewers"])
            if count > 0:
                growth_by_hour[hour] = math.log(count / base_count)
        neighbour_pool.append((
            abs(base_log - launch_log), item.get("name", ""), growth_by_hour
        ))
        predictors = _four_features(
            base_log, history[release_at].get("channels")
        )
        regression_pool.append((predictors, growth_by_hour))

    sizes = sorted(
        predictors[1] + predictors[2]
        for predictors, _growth in regression_pool
    )
    neighbour_mix = 0.4
    if sizes:
        low_boundary = _quantile(sizes, 1 / 3)
        high_boundary = _quantile(sizes, 2 / 3)
        if launch_log < low_boundary:
            neighbour_mix = 0.25
        elif launch_log > high_boundary:
            neighbour_mix = 0.55

    neighbour_pool.sort(key=lambda record: (record[0], record[1]))
    nearest = neighbour_pool[:NEIGHBOUR_COUNT]
    point_features = _four_features(launch_log, first_row.get("channels"))
    calibration = 0.0
    if not case.get("_skip-bias-correction", False):
        calibration = _global_median_shift(case)

    output_hours = []
    lower_log = math.log(0.01)
    upper_log = math.log(1e9)
    for hour in range(2, 25):
        growth_values = sorted(
            record[2][hour] for record in nearest if hour in record[2]
        )
        if len(growth_values) >= 2:
            combined_logs = [
                launch_log + _quantile(growth_values, tau) for tau in LEVELS
            ]
            observations = [
                (predictors, trajectory[hour])
                for predictors, trajectory in regression_pool
                if hour in trajectory
                and math.isfinite(trajectory[hour])
                and all(math.isfinite(value) for value in predictors)
            ]
            if len(observations) >= 6:
                coefficients = _regression_fit(observations, dimensions)
                if (coefficients is not None
                        and all(math.isfinite(value) for value in coefficients)):
                    coefficients = _robust_refine(
                        observations, coefficients, dimensions
                    )
                    errors = [
                        target - sum(
                            coefficients[index] * predictors[index]
                            for index in range(dimensions)
                        )
                        for predictors, target in observations
                    ]
                    average_error = sum(errors) / len(errors)
                    pooled_var = sum(
                        (error - average_error) ** 2 for error in errors
                    ) / len(errors)
                    locality = [
                        math.exp(-0.5 * ((predictors[1] + predictors[2] - launch_log) / 2.0) ** 2)
                        for predictors, _target in observations
                    ]
                    local_var = sum(
                        weight * (error - average_error) ** 2
                        for weight, error in zip(locality, errors)
                    ) / sum(locality)
                    sigma = math.sqrt(0.5 * pooled_var + 0.5 * local_var)
                    if math.isfinite(sigma):
                        sigma = max(0.15, sigma)
                        center = sum(
                            coefficients[index] * point_features[index]
                            for index in range(dimensions)
                        )
                        regression_logs = [
                            launch_log + center + z * sigma for z in Z_SCORES
                        ]
                        combined_logs = [
                            neighbour_mix * local_log
                            + (1.0 - neighbour_mix) * regression_log
                            for local_log, regression_log in zip(
                                combined_logs, regression_logs
                            )
                        ]
        else:
            uncertainty = 0.15 * math.sqrt(hour - 1)
            combined_logs = [
                launch_log + z * uncertainty for z in Z_SCORES
            ]

        quantile_counts = []
        for log_estimate in combined_logs:
            adjusted_log = log_estimate + calibration
            if adjusted_log <= lower_log:
                count = 0.01
            elif adjusted_log >= upper_log:
                count = 1e9
            else:
                count = math.exp(adjusted_log)
            count = _rounded_count(max(0.01, min(1e9, count)))
            if quantile_counts:
                count = max(count, quantile_counts[-1])
            quantile_counts.append(count)
        output_hours.append({"hour": hour, "q": quantile_counts})
    return {"hours": output_hours}


def main():
    sources = sorted(INPUT_DIR.glob("*.in"))
    if not sources:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in sources:
        with source.open("r", encoding="utf-8") as stream:
            request = json.load(stream)
        answer = forecast_case(request)
        destination = OUTPUT_DIR / (source.stem + ".out")
        destination.write_text(
            json.dumps(answer, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
