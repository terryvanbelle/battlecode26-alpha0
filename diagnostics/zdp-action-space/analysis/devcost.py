"""Cost per executed deviation at floor 0.2 (no games; reads cpi-x0-summary.json and devfb-cpi-x0.json).

Per point: decisions per team-game (cpi-x0 rows; frozen points are a 5% sample), the deviations a floor of 0.2 adds
(0.2 (1 - 1/m) plus the net's own softmax deviations), the share that executes rather than falls back (fb_given_dev),
the sensitivity loss (0.5 - score of zdph<p> vs zdpg39), loss per executed deviation, and loss / sqrt(executed): the
per-game effect a deviation pattern of that size has relative to its count (an outcome-level gradient signal scales as
the per-deviation effect times sqrt(count), so a fixed exploration cost C buys signal ~ C / sqrt(count)).
"""
import json, math
s = json.load(open('cpi-x0-summary.json')); dv = json.load(open('devfb-cpi-x0.json'))
sens = {1: .27, 2: .22, 3: .28, 4: .42, 5: .50, 6: .41, 7: .44, 8: .44, 10: .53, 11: .27, 12: .44, 13: .375, 14: .30, 16: .48}
TG = 2 * s['games']
print(f"{'pt':>3} {'dec/tg':>7} {'m':>5} {'fb|dev':>6} {'dev@.2':>7} {'exec@.2':>7} {'loss':>6} {'loss/exec':>9} {'C/sqrtN':>8}")
rows = []
for p, sc in sens.items():
    P = s['points'][str(p)]; D = dv.get(str(p), {})
    dec = P['rows'] / (1 if P['trainable'] else s['frozen_sample']) / TG
    m = D.get('mean_allowed') or len(P['option_hist'])
    fb = D.get('fb_given_dev', 0.0) or 0.0
    soft_dev = max(0.0, (D.get('dev_rate', 0) or 0) - P['floor'] * (1 - 1 / m))
    dev02 = dec * (0.2 * (1 - 1 / m) + soft_dev * 0.8)
    ex02 = dev02 * (1 - fb)
    loss = 0.5 - sc
    rows.append((p, dec, m, fb, dev02, ex02, loss, loss / ex02 if ex02 else 0, loss / math.sqrt(ex02) if ex02 else 0))
for r in sorted(rows, key=lambda r: -r[1]):
    print(f"{r[0]:>3} {r[1]:7.0f} {r[2]:5.1f} {r[3]:6.2f} {r[4]:7.0f} {r[5]:7.0f} {r[6]:6.3f} {r[7]:9.5f} {r[8]:8.4f}")
