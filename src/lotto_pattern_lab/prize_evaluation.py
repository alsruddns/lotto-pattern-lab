"""Separate first-prize and second-prize hit accounting for paired predictions."""

from collections.abc import Mapping, Sequence

from .dataset import DrawDataset
from .draw import Draw


def classify_prize_tier(
    predicted_numbers: Sequence[int],
    winning_numbers: Sequence[int],
    bonus_number: int | None,
) -> dict[str, int | bool | str | None]:
    """Classify one six-number combination using the official 6/45 tier rules."""
    predicted = Draw.from_numbers(tuple(predicted_numbers))
    actual = Draw.from_numbers(tuple(winning_numbers), bonus_number)
    matches = len(set(predicted.numbers) & set(actual.numbers))
    bonus_included = None if actual.bonus is None else actual.bonus in predicted.numbers
    if matches == 6:
        tier = 1
        status = "판정 가능"
    elif matches == 5 and actual.bonus is None:
        tier = None
        status = "보너스 번호 미수집"
    elif matches == 5:
        tier = 2 if bonus_included else 3
        status = "판정 가능"
    elif matches == 4:
        tier, status = 4, "판정 가능"
    elif matches == 3:
        tier, status = 5, "판정 가능"
    else:
        tier, status = None, "당첨 등수 없음"
    return {
        "main_match_count": matches,
        "bonus_included": bonus_included,
        "prize_tier": tier,
        "prize_tier_status": status,
    }


def evaluate_combo_prize_predictions(
    predictions: Mapping[int, Sequence[int]],
    actual_draws: Mapping[int, Draw],
) -> dict[str, object]:
    """Count exact prize tiers for saved six-number predictions, without guessing missing data."""
    tier_hits = {str(tier): 0 for tier in range(1, 6)}
    evaluated = 0
    known_bonus = 0
    unknown_bonus = 0
    missing_actual = 0
    for round_number, predicted in predictions.items():
        actual = actual_draws.get(int(round_number))
        if actual is None:
            missing_actual += 1
            continue
        result = classify_prize_tier(predicted, actual.numbers, actual.bonus)
        evaluated += 1
        if actual.bonus is None:
            unknown_bonus += 1
        else:
            known_bonus += 1
        tier = result["prize_tier"]
        if tier is not None:
            tier_hits[str(tier)] += 1
    return {
        "prediction_count": len(predictions),
        "evaluated_prediction_count": evaluated,
        "missing_actual_count": missing_actual,
        "prize_tier_hits": tier_hits,
        "second_prize_hits": tier_hits["2"] if known_bonus else None,
        "second_prize_denominator": known_bonus,
        "second_prize_unknown_bonus_count": unknown_bonus,
        "second_prize_status": (
            ("미수집" if unknown_bonus else "판정 불가")
            if known_bonus == 0
            else "판정 가능"
            if unknown_bonus == 0
            else "부분 집계"
        ),
        "counting_unit": "활성 예측 조합 수",
    }


def evaluate_prize_predictions(
    dataset: DrawDataset,
    main_predictions: Mapping[int, Sequence[int]],
    bonus_predictions: Mapping[int, int | None],
) -> dict[str, object]:
    """Count tiers from six-number predictions; bonus-model output is not a prize input.

    `bonus_predictions` remains accepted for call compatibility, but the official second
    prize is determined only by whether the actual bonus is one of the predicted six numbers.
    """
    records = {record.round_number: record for record in dataset.records}
    # Bonus-model output is retained for source compatibility only. Official prize
    # evaluation is based on each saved six-number main prediction and actual draw.
    rounds = sorted(set(main_predictions) & set(records))
    if not rounds:
        return {
            "evaluation_round_count": 0,
            "first_prize_exact_six": {"hits": 0, "denominator": 0},
            "second_prize_five_plus_bonus": {
                "hits": None,
                "denominator": 0,
                "unknown_bonus_rounds": 0,
                "status": "미수집",
                "counting_unit": "활성 예측 조합 수",
            },
        }
    first_hits = 0
    second_hits = 0
    known_bonus_rounds = 0
    unknown_bonus_rounds = 0
    for round_number in rounds:
        actual = records[round_number].draw
        predicted_main = tuple(main_predictions[round_number])
        if not predicted_main or len(predicted_main) != 6 or len(set(predicted_main)) != 6:
            raise ValueError(
                f"Round {round_number} main prediction must contain six unique numbers."
            )
        classification = classify_prize_tier(predicted_main, actual.numbers, actual.bonus)
        matches = int(classification["main_match_count"])
        first_hits += matches == 6
        if actual.bonus is None:
            unknown_bonus_rounds += 1
        else:
            known_bonus_rounds += 1
            second_hits += matches == 5 and actual.bonus in predicted_main
    second_status = (
        "미수집"
        if known_bonus_rounds == 0 and unknown_bonus_rounds
        else "부분 집계"
        if unknown_bonus_rounds
        else "판정 가능"
    )
    return {
        "evaluation_round_count": len(rounds),
        "target_rounds": rounds,
        "first_prize_exact_six": {"hits": first_hits, "denominator": len(rounds)},
        "second_prize_five_plus_bonus": {
            "hits": second_hits if known_bonus_rounds else None,
            "denominator": known_bonus_rounds,
            "unknown_bonus_rounds": unknown_bonus_rounds,
            "status": second_status,
            "counting_unit": "예측 조합 수",
        },
        "same_round_inputs_excluded_by_feature_contract": True,
    }
