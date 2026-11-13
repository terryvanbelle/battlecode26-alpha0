"""Constants mirrored from java/student/alpha0/student/{Obs,Actions,Records}.java."""
OBS_VERSION = 'student-obs-v2'
L, CL, G, CG, NS = 11, 16, 16, 8, 48
LOCAL, GLOBAL = L * L * CL, G * G * CG
OBS = LOCAL + GLOBAL + NS                       # 4032 (obs-v2)
TC = 49
NT = 11
TARGET0 = 17
THROW = TARGET0 + NT * TC
N_ACTIONS = THROW + 1 + 8 + 1                   # 566
MASK_BYTES = (N_ACTIONS + 63) // 64 * 8         # 72
HEADER = 16
RECORD = HEADER + OBS + MASK_BYTES
TYPES = ['attack', 'carry', 'build', 'rat_trap', 'cat_trap', 'place_dirt', 'remove_dirt', 'pickup', 'transfer',
         'remove_rat_trap', 'remove_cat_trap']


def action_name(a: int) -> str:
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    if a == 0:
        return 'end'
    if a < 9:
        return 'move_' + dirs[a - 1]
    if a < 17:
        return 'turn_' + dirs[a - 9]
    if a < THROW:
        t, c = divmod(a - TARGET0, TC)
        return f'{TYPES[t]}({c % 7 - 3},{c // 7 - 3})'
    if a == THROW:
        return 'throw'
    if a < THROW + 9:
        return 'drop_' + dirs[a - THROW - 1]
    return 'become_king'


def action_group(a: int) -> str:
    n = action_name(a)
    return n.split('(')[0].split('_')[0] if not n.startswith(('rat_', 'cat_', 'place_', 'remove_')) else n.split('(')[0]


def group_index(a: int) -> int:
    """Mirror of Net.group (Java): END, MOVE, TURN, 11 target types, THROW, DROP, BECOME_KING."""
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
    if a < THROW + 9:
        return 4 + NT
    return 5 + NT


GROUP_OF = [group_index(a) for a in range(N_ACTIONS)]
N_GROUPS = 6 + NT
