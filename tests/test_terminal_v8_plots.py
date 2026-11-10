"""Broader-map plotting never turns incomplete evidence into a successful result."""
from copy import deepcopy
import json
import pytest
from alpha0 import terminal_v8_common as C
from alpha0 import terminal_v8_report as R
from scripts import plot_terminal_v8_development as P

@pytest.fixture
def empty(tmp_path):
    config=C.load_config();identity=C.identity(config)
    return R.report(tmp_path/'run',tmp_path/'progress',config,identity)

def test_empty_phase_shows_coverage_without_scores(empty,tmp_path):
    P.generate(empty,tmp_path)
    assert (tmp_path/'comparison.png').read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    assert (tmp_path/'coverage.png').is_file()
    text=(tmp_path/'README.md').read_text()
    assert '0/768' in text and 'No automatic promotion' in text
    assert empty['benchmarks']['panels']['cp2']['mean_score'] is None

def test_missing_games_cannot_claim_improvement(empty,tmp_path):
    empty['decision']['criteria_pass']=True
    with pytest.raises(ValueError,match='incomplete qualifying'):P.generate(empty,tmp_path)

def test_counts_and_temperatures_are_checked(empty,tmp_path):
    for mutation in ('counts','temperature','old_data'):
        altered=deepcopy(empty)
        if mutation=='counts':altered['benchmarks']['panels']['terminal_head']['wins']=1
        elif mutation=='temperature':altered['sampling']['continuation_temperature']=1
        else:altered['sampling']['earlier_trajectory_data_reused']=True
        with pytest.raises(ValueError):P.generate(altered,tmp_path)
