import copy
import json
from pathlib import Path
import pytest
from alpha0.imitation.terminal import validate_terminal
from alpha0.imitation.trace import digest

ROOT=Path(__file__).resolve().parents[1]

def pair():
    root=ROOT/'diagnostics/zdb-imitation-planning/world-audit/terminal-example'
    return json.loads((root/'terminal-proof.json').read_text()),json.loads((root/'world-audit.json').read_text())

def test_actual_saved_terminal():
    p,w=pair();assert validate_terminal(p,w)=={'terminal_verified':True,'training_approved':False,'journal_complete':False}

@pytest.mark.parametrize('field,value',[('running',True),('currentRound',99)])
def test_rehashed_nonterminal_or_wrong_round(field,value):
    p,w=pair();w['world']['fields']['battlecode.world.GameWorld.'+field]=value;p['physical_state_sha256']=digest(w)
    with pytest.raises(ValueError,match='not terminal'):validate_terminal(p,w)

def test_forged_winner_and_training_flag():
    p,w=pair();p['winner']='A'
    with pytest.raises(ValueError,match='result mismatch'):validate_terminal(p,w)
    p,w=pair();p['training_approved']=True
    with pytest.raises(ValueError,match='scope'):validate_terminal(p,w)
