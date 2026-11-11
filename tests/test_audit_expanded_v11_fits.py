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
