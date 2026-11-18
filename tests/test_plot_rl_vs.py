"""scripts/plot_rl_vs.py: the switch from v0 to the newest pool version once the learner wins every game against v0 in
an iteration, the status sentence scripts/rl_chart_sync.sh writes from it, and the chart drawn with it."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import plot_rl_vs as pv  # noqa: E402


def progress(tmp_path, scores):
    """A progress.jsonl with one row per (iteration, {opponent: (points, games)})."""
    f = tmp_path / 'progress.jsonl'
    f.write_text(''.join(json.dumps({'iter': k, 'score': {o: {'points': p, 'games': g} for o, (p, g) in sc.items()}})
                         + '\n' for k, sc in scores))
    return f


SCORES = [
    (0, {}),                                         # iteration 0 plays only itself
    (1, {'v0': (5, 10)}),
    (2, {'v0': (4, 4)}),                             # every game won, but too few, and iteration 1 lost some
    (3, {'v0': (9, 9), 'v2': (3, 8)}),               # every game won: the switch
    (4, {'v0': (8, 8), 'v2': (5, 9), 'v3': (6, 10)}),
    (13, {'v0': (7, 8), 'v12': (2, 6), 'v4': (5, 9)}),   # newest by number, not by text
]


def test_switches_after_the_first_perfect_iteration(tmp_path):
    before, after, switch = pv.newest_after_perfect(progress(tmp_path, SCORES), 'v0')
    assert switch == 3
    assert before == [(1, 5, 10), (2, 4, 4), (3, 9, 9)]
    assert after == [(4, 6, 10, 'v3'), (13, 2, 6, 'v12')]


def test_perfect_run_counts_back_to_enough_games(tmp_path):
    rows = [(1, 5, 10), (2, 4, 4), (3, 6, 6)]
    assert pv.perfect_run(rows, 8) == [(2, 4, 4), (3, 6, 6)]
    assert pv.perfect_run(rows, 11) is None                  # reaches iteration 1, which lost games
    assert pv.perfect_run(rows[:2], 8) is None               # all won, but only 4 games
    assert pv.perfect_run([(1, 9.5, 10)], 8) is None         # a draw is not a win


def test_switch_over_several_small_iterations(tmp_path):
    scores = [(10, {'v0': (5, 6)}), (11, {'v0': (6, 6)}), (12, {'v0': (6, 6), 'v10': (4, 8)}), (13, {'v12': (3, 6)})]
    before, after, switch = pv.newest_after_perfect(progress(tmp_path, scores), 'v0')
    assert (switch, after) == (12, [(13, 3, 6, 'v12')])
    assert pv.pool_status(progress(tmp_path, scores)) == (
        'Pool games against v0, iterations 10–12: 17/18 = 0.94; the learner won all 12 in iterations 11–12, so the chart '
        'now compares against the newest pool version: iteration 13 against v12: 3/6')


def test_no_switch_without_a_perfect_iteration(tmp_path):
    before, after, switch = pv.newest_after_perfect(progress(tmp_path, SCORES[:3]), 'v0')
    assert (switch, after) == (None, [])
    assert before == pv.pool_series(progress(tmp_path, SCORES[:3]), 'v0')


def test_last_row_per_iteration_wins(tmp_path):
    f = progress(tmp_path, SCORES[:2])
    f.write_text(f.read_text() + json.dumps({'iter': 1, 'score': {'v0': {'points': 7, 'games': 10}}}) + '\n\n')
    assert pv.pool_series(f, 'v0') == [(1, 7, 10)]


def test_status_sentence(tmp_path):
    assert pv.pool_status(progress(tmp_path, SCORES[:3])) == 'Pool games against v0, iterations 1–2: 9/14 = 0.64'
    assert pv.pool_status(progress(tmp_path, SCORES[:4])) == (
        'Pool games against v0, iterations 1–3: 18/23 = 0.78; the learner won all 9 in iteration 3, so the chart now '
        'compares against the newest pool version')
    assert pv.pool_status(progress(tmp_path, SCORES)) == (
        'Pool games against v0, iterations 1–3: 18/23 = 0.78; the learner won all 9 in iteration 3, so the chart now '
        'compares against the newest pool version: iterations 4–13 8/16 = 0.50 (iteration 13 against v12: 2/6)')
    assert pv.pool_status(progress(tmp_path, SCORES[:1])) == ''


def test_chart_after_the_switch(tmp_path):
    gates = tmp_path / 'gates.jsonl'
    gates.write_text(json.dumps({'mode': 'anchor-zdb', 'version': 3, 'points': 1, 'games': 32}) + '\n')
    out = tmp_path / 'c.svg'
    pv.main([str(progress(tmp_path, SCORES)), '--out', str(out), '--gates', str(gates), '--newest-after-perfect'])
    svg = out.read_text()
    assert 'v0 100% at 3; then the newest version' in svg                     # the switch mark
    assert 'v13 vs v12: 2/6' in svg and 'v4 vs v3: 6/10' in svg                # hover names the opponent
    assert 'vs v0 (pool games): pooled 18/23' in svg
    assert 'vs the newest pool version (pool games): pooled 8/16' in svg
    assert 'anchor-zdb gate: pooled 1/32' in svg
    assert svg.count('class="dot2"') == 3 and svg.count('class="dot1"') == 2  # legend dot + one per point


def test_series_keep_their_colours(tmp_path):
    """The gate series stays orange (slot 1) when the pool series has nothing to draw."""
    gates = tmp_path / 'gates.jsonl'
    gates.write_text(json.dumps({'mode': 'anchor-zdb', 'version': 3, 'points': 1, 'games': 32}) + '\n')
    out = tmp_path / 'c.svg'
    pv.main([str(progress(tmp_path, SCORES[:1])), '--out', str(out), '--gates', str(gates)])
    svg = out.read_text()
    assert 'class="dot1"' in svg and 'class="dot0"' not in svg
