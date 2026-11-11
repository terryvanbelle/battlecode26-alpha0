from copy import deepcopy
import pytest
from alpha0 import expanded_v11_fit as F

def test_fixed_fit_protocol_and_no_partial_pair():
    config={'fit':deepcopy(F.FIT),'torch_threads':1,'experiment_id':'expanded-terminal-v11','selection_seeds':[223003,223013]}
    F.validate_config(config)
    for key,value in [('updates',2048),('anchor_kl_weight',0),('seed',205101)]:
        bad=deepcopy(config);bad['fit'][key]=value
        with pytest.raises(ValueError):F.validate_config(bad)
    with pytest.raises(ValueError):F.verify_complete({'complete':False},{},None)
