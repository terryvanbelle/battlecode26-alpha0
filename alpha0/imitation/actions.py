"""Versioned primitive targets for the new student; not macro_v2 weights.

Argument tokens preserve signed int32 values exactly via four byte tokens.
Location arguments are relative to the acting robot. END_TURN is a real target,
not an invented idle action. This codec does not certify legality or training data.
"""
from .trace import OPERATIONS, DIRECTIONS, arguments

VERSION='primitive-byte-arguments-v1'
OPERATIONS_ORDER=tuple(sorted(OPERATIONS))+('END_TURN',)
DIRECTION_ORDER=('NORTH','NORTHEAST','EAST','SOUTHEAST','SOUTH','SOUTHWEST','WEST','NORTHWEST','CENTER')
EOS=256

def _integer(value):
    if type(value) is not int or not -(2**31)<=value<2**31:raise ValueError('int32 required')
    return list(value.to_bytes(4,'big',signed=True))

def encode(operation,args,origin):
    if operation=='END_TURN':
        if args:raise ValueError('END_TURN has no arguments')
        return OPERATIONS_ORDER.index(operation),[EOS]
    arguments(operation,args)
    tokens=[]
    for kind,value in zip(OPERATIONS[operation],args):
        if kind=='direction':tokens.append(DIRECTION_ORDER.index(value))
        elif kind=='int':tokens.extend(_integer(value))
        else:
            if not isinstance(origin,(tuple,list)) or len(origin)!=2:raise ValueError('origin required')
            tokens.extend(_integer(value[0]-origin[0]));tokens.extend(_integer(value[1]-origin[1]))
    return OPERATIONS_ORDER.index(operation),tokens+[EOS]

def decode(operation_id,tokens,origin):
    if type(operation_id) is not int or not 0<=operation_id<len(OPERATIONS_ORDER):raise ValueError('operation id')
    if not isinstance(tokens,list) or not tokens or tokens[-1]!=EOS or any(type(t) is not int or not 0<=t<256 for t in tokens[:-1]):raise ValueError('argument tokens')
    op=OPERATIONS_ORDER[operation_id];body=tokens[:-1];position=0;args=[]
    def number():
        nonlocal position
        if position+4>len(body):raise ValueError('truncated integer')
        n=int.from_bytes(bytes(body[position:position+4]),'big',signed=True);position+=4;return n
    for kind in (() if op=='END_TURN' else OPERATIONS[op]):
        if kind=='direction':
            if position>=len(body) or body[position]>=len(DIRECTION_ORDER):raise ValueError('direction token')
            args.append(DIRECTION_ORDER[body[position]]);position+=1
        elif kind=='int':args.append(number())
        else:
            if not isinstance(origin,(tuple,list)) or len(origin)!=2:raise ValueError('origin required')
            args.append([number()+origin[0],number()+origin[1]])
    if position!=len(body):raise ValueError('extra argument tokens')
    if op!='END_TURN':arguments(op,args)
    return op,args
