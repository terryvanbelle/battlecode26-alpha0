"""Independent checks for the read-only prequential value diagnostic."""
import math

import numpy as np
import pytest

from scripts.audit_value import equal_game_summary, opponent_group_summaries, score_game_logits, targets_from_players


def test_terminal_labels_follow_the_acting_player():
    # Both teams in the same winning-A game have opposite targets. Reversing
    # the outcome must penalize the very same forecasts, not silently reuse A.
    logits = np.array([math.log(4), -math.log(4)])
    np.testing.assert_array_equal(targets_from_players([0, 1], 1), [1, 0])
    np.testing.assert_array_equal(targets_from_players([0, 1], -1), [0, 1])
    np.testing.assert_array_equal(targets_from_players([0, 1], 0), [.5, .5])
    correct = score_game_logits(logits, [0, 1], 1)
    inverted = score_game_logits(logits, [0, 1], -1)
    assert correct["brier"] == pytest.approx(.04)
    assert correct["value_mse"] == pytest.approx(.16)
    assert correct["bce"] == pytest.approx(-math.log(.8))
    assert inverted["brier"] == pytest.approx(.64)
    with pytest.raises(ValueError, match="team 0/1"):
        targets_from_players([0, 2], 1)


def test_equal_game_weight_survives_unequal_retained_lengths_and_bins():
    # One short wrong game and one long correct game receive equal weight.
    # Position-weighting would hide the short game's error (.104 vs .34).
    short = score_game_logits([math.log(4)], [0], -1)
    long = score_game_logits([-math.log(4)] * 9, [0] * 9, -1)
    summary = equal_game_summary([short, long])
    assert summary["brier"] == pytest.approx((.64 + .04) / 2)
    assert summary["brier"] != pytest.approx((.64 + 9 * .04) / 10)
    occupied = [row for row in summary["reliability_bins"] if row["equal_game_weight"]]
    assert [row["equal_game_weight"] for row in occupied] == pytest.approx([.5, .5])
    assert [row["mean_prediction"] for row in occupied] == pytest.approx([.2, .8])
    assert [row["observed_win_rate"] for row in occupied] == [0, 0]
    assert summary["reliability_ece"] == pytest.approx(.5)


def test_extreme_wrong_logits_are_finite_and_endpoint_bins_are_retained():
    scored = score_game_logits([-1000, 1000], [0, 1], 1)
    assert scored["brier"] == 1
    assert scored["value_mse"] == 4
    assert scored["bce"] == 1000
    assert scored["saturated_fraction"] == 1
    assert scored["reliability_bins"][0]["positions"] == 1
    assert scored["reliability_bins"][-1]["positions"] == 1


def test_league_labels_are_reported_separately_from_both_team_targets():
    same = {"opponent_checkpoint": None, **score_game_logits([0, 0], [0, 1], 1)}
    league = {"opponent_checkpoint": "checkpoint-0000.pt", **score_game_logits([0] * 8, [1] * 8, -1)}
    groups = {row["opponent_group"]: row for row in opponent_group_summaries([same, league])}
    assert groups["same_policy"]["mean_target_win_rate"] == .5
    assert groups["historical_opponent"]["mean_target_win_rate"] == 1
    assert equal_game_summary([same, league])["mean_target_win_rate"] == .75
