"""Constants mirrored from java/student/alpha0/student/{Obs,Actions,Records}.java (obs-v4 / actions-v3 / records-v2).
Game.java writes obs_size, n_actions and record_size into result.json; train.py checks them against these."""
OBS_VERSION = 'student-obs-v4'
ACTION_VERSION = 'student-actions-v3'
L, CL, G, CG = 11, 25, 16, 8
NS, SQ, SQM = 96, 16, 8
LOCAL, GLOBAL = L * L * CL, G * G * CG
SCALARS = LOCAL + GLOBAL
DENSE = SCALARS + NS + SQ * SQM                 # 5297
MAP_MAX, MAP_PLANES = 60, 8
MAP_BITS = MAP_MAX * MAP_MAX * MAP_PLANES       # 28800 (bit offset of the shared array)
BITS = MAP_BITS + 64 * 10 + 32 * 10 + SQ * 32   # 30272
OTHER_BITS = BITS - MAP_BITS
BIT_BYTES = (BITS + 7) // 8
OBS = DENSE + BIT_BYTES                         # 9081
TC = 49
NT = 11
TARGET0 = 17
THROW = TARGET0 + NT * TC                       # 556
DROP0 = THROW + 1
BECOME_KING = DROP0 + 8
SQUEAK = BECOME_KING + 1
WRITE_SHARED0 = SQUEAK + 1
WRITE_REG0 = WRITE_SHARED0 + 64
DISINTEGRATE = WRITE_REG0 + 32
N_ACTIONS = DISINTEGRATE + 1                    # 664 (resign() is deliberately not an action)
LEGACY_N_ACTIONS = (665,)                       # actions-v2 records: same layout plus a trailing RESIGN index (marked
                                                # legal in teacher masks, never a label: zdb never resigns)
MASK_BYTES = (N_ACTIONS + 63) // 64 * 8         # 88
HEADER = 24
RECORD = HEADER + OBS + MASK_BYTES
PARAM_MAX_BITS = 32
P_NONE, P_AMOUNT, P_SQUEAK, P_VALUE = 0, 1, 2, 3
PARAM_BITS = [0, 16, 32, 10]
ATTACK, PICKUP, TRANSFER = 0, 7, 8
TYPES = ['attack', 'carry', 'build', 'rat_trap', 'cat_trap', 'place_dirt', 'remove_dirt', 'pickup', 'transfer',
         'remove_rat_trap', 'remove_cat_trap']
DIRS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def param_kind(a: int) -> int:
    if TARGET0 <= a < THROW:
        return P_AMOUNT if (a - TARGET0) // TC in (ATTACK, PICKUP, TRANSFER) else P_NONE
    if a == SQUEAK:
        return P_SQUEAK
    if WRITE_SHARED0 <= a < DISINTEGRATE:
        return P_VALUE
    return P_NONE


def action_name(a: int) -> str:
    if a == 0:
        return 'end'
    if a < 9:
        return 'move_' + DIRS[a - 1]
    if a < 17:
        return 'turn_' + DIRS[a - 9]
    if a < THROW:
        t, c = divmod(a - TARGET0, TC)
        return f'{TYPES[t]}({c % 7 - 3},{c // 7 - 3})'
    if a == THROW:
        return 'throw'
    if a < BECOME_KING:
        return 'drop_' + DIRS[a - DROP0]
    if a == BECOME_KING:
        return 'become_king'
    if a == SQUEAK:
        return 'squeak'
    if a < WRITE_REG0:
        return f'write_shared[{a - WRITE_SHARED0}]'
    if a < DISINTEGRATE:
        return f'write_reg[{a - WRITE_REG0}]'
    return 'disintegrate'


def action_group(a: int) -> str:
    n = action_name(a)
    if n.startswith(('write_shared', 'write_reg')):
        return n.split('[')[0]
    return n.split('(')[0].split('_')[0] if not n.startswith(('rat_', 'cat_', 'place_', 'remove_')) else n.split('(')[0]


def group_index(a: int) -> int:
    """Mirror of Net.group (Java)."""
    if a == 0:
        return 0
    if a < 9:
        return 1
    if a < TARGET0:
        return 2
    if a < THROW:
        return 3 + (a - TARGET0) // TC
    if a == THROW:
        return 3 + NT
    if a < BECOME_KING:
        return 4 + NT
    if a == BECOME_KING:
        return 5 + NT
    if a == SQUEAK:
        return 6 + NT
    if a < WRITE_REG0:
        return 7 + NT
    if a < DISINTEGRATE:
        return 8 + NT
    return 9 + NT


GROUP_OF = [group_index(a) for a in range(N_ACTIONS)]
N_GROUPS = 10 + NT
PARAM_KIND_OF = [param_kind(a) for a in range(N_ACTIONS)]
