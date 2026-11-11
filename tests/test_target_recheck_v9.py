import pytest
from scripts.run_target_recheck_v9 import paired_decision

def rows(value):return [{'complete':True,'source_json':str(i//2),'fresh_seed_advantage':value} for i in range(32)]
def test_fixed_denominator_and_source_clusters():
    zero=paired_decision(rows(0));assert zero['mean_advantage']==0 and not zero['review_for_expansion']
    positive=paired_decision(rows(.25));assert positive['source_cluster_bootstrap_95']==[.25,.25] and positive['review_for_expansion']
    assert not paired_decision(rows(.05))['review_for_expansion']
    with pytest.raises(AssertionError):paired_decision(rows(.25)[:-1])
    incomplete=rows(.25);incomplete[0]['complete']=False
    with pytest.raises(AssertionError):paired_decision(incomplete)
def test_opposing_source_clusters_do_not_establish_gain():
    r=rows(.25)
    for x in r[:16]:x['fresh_seed_advantage']=-.25
    out=paired_decision(r);assert out['mean_advantage']==0 and out['source_cluster_bootstrap_95'][0]<0<out['source_cluster_bootstrap_95'][1]
    assert not out['review_for_expansion']
