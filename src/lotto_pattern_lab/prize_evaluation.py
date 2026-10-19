"""Separate first-prize and second-prize hit accounting for paired predictions."""

from collections.abc import Mapping, Sequence

from .dataset import DrawDataset


def evaluate_prize_predictions(
    dataset: DrawDataset,
    main_predictions: Mapping[int, Sequence[int]],
    bonus_predictions: Mapping[int, int | None],
) -> dict[str, object]:
    """Count exact six and exactly five plus bonus on shared, explicit target rounds."""
    records = {record.round_number: record for record in dataset.records}
    rounds = sorted(set(main_predictions) & set(bonus_predictions) & set(records))
    if not rounds:
        return {
            "evaluation_round_count": 0,
            "first_prize_exact_six": {"hits": 0, "denominator": 0},
            "second_prize_five_plus_bonus": {"hits": 0, "denominator": 0},
        }
    first_hits = 0
    second_hits = 0
    for round_number in rounds:
        actual = records[round_number].draw
        if actual.bonus is None:
            raise ValueError(f"Draw {round_number} has no bonus number.")
        predicted_main = tuple(main_predictions[round_number])
        if predicted_main and (len(predicted_main) != 6 or len(set(predicted_main)) != 6):
            raise ValueError(
                f"Round {round_number} main prediction must be empty or six unique numbers."
            )
        matches = len(set(predicted_main) & set(actual.numbers)) if predicted_main else 0
        first_hits += matches == 6
        second_hits += matches == 5 and bonus_predictions[round_number] == actual.bonus
    return {
        "evaluation_round_count": len(rounds),
        "target_rounds": rounds,
        "first_prize_exact_six": {"hits": first_hits, "denominator": len(rounds)},
        "second_prize_five_plus_bonus": {"hits": second_hits, "denominator": len(rounds)},
        "same_round_inputs_excluded_by_feature_contract": True,
    }
