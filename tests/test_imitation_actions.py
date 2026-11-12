import pytest
from alpha0.imitation.actions import encode,decode,OPERATIONS_ORDER,EOS
from alpha0.imitation.trace import OPERATIONS

@pytest.mark.parametrize('op',OPERATIONS_ORDER)
def test_all_operations_round_trip(op):
    kinds=OPERATIONS.get(op,())
    args=[('SOUTHWEST' if k=='direction' else -2147483648 if k=='int' else [59,0]) for k in kinds]
    operation,tokens=encode(op,args,[3,7])
    assert decode(operation,tokens,[3,7])==(op,args)

def test_translation_and_end_turn():
    assert encode('attack',[[10,12]],[8,9])==encode('attack',[[20,32]],[18,29])
    assert decode(*encode('END_TURN',[],None),None)==('END_TURN',[])

@pytest.mark.parametrize('tokens',[[EOS,EOS],[],[0,EOS],[True,0,0,0,EOS],[0,0,0,0,0,EOS]])
def test_invalid_integer_tokens(tokens):
    op=OPERATIONS_ORDER.index('squeak')
    with pytest.raises(ValueError):decode(op,tokens,[0,0])

def test_signed_message_extremes():
    for n in (-2**31,-1,0,2**31-1):
        assert decode(*encode('squeak',[n],[0,0]),[0,0])==('squeak',[n])
