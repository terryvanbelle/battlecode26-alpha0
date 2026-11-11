import numpy as np
import pytest
from alpha0.terminal_v8_fit import make_batch_schedule
from scripts.audit_expanded_v11_fits import verify_batches


def test_independent_full_batchstream_and_corruption():
    # Covers all64 fixed slots and1024 anchors, without any fitting or games.
    d={'root_strata':np.array([[m,r,t] for m in range(8) for r in range(2) for t in range(2) for _ in range(2)]),
       'anchor_game':np.repeat(np.arange(64),16),
       'anchor_player':np.tile(np.repeat([0,1],8),64)}
    generated=make_batch_schedule(d['root_strata'],d['anchor_game'],d['anchor_player'],{'updates':4096,'batch_size':32,'anchor_batch_size':32,'seed':223101})
    batches={k:generated[k] for k in ('root','anchor')}
    assert verify_batches(d,batches).bit_generator.state==generated['rng_state']
    changed={k:v.copy() for k,v in batches.items()};changed['root'][4095,31]=(changed['root'][4095,31]+1)%64
    with pytest.raises(AssertionError):verify_batches(d,changed)
    changed={k:v[:2048] for k,v in batches.items()}
    with pytest.raises(AssertionError):verify_batches(d,changed)


def test_conflicting_identical_inputs_preserve_slot_denominator():
    from scripts.audit_expanded_v11_fits import target_ambiguity
    result=target_ambiguity(['a','a','a','b'],[1,1,2,3])
    assert result['slots']==4 and result['unique_inputs']==2
    assert result['maximum_consistent_label_matches']==3
    assert result['conflicting_input_groups'][0]['slot_indices']==[0,1,2]
    assert target_ambiguity(['a','a','b'],[1,1,3])['maximum_consistent_label_matches']==3
    with pytest.raises(ValueError):target_ambiguity(['a','b'],[1])
