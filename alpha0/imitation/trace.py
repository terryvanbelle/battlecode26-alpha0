"""Strict, bounded primitive-call trace format and complete-pair equivalence gate.

Records describe calls to the public controller API; they are NOT student tensor
inputs or accepted training examples. Engine integration must establish actual
public-API provenance. Self-declared records alone cannot certify legal sensing.
"""
import hashlib,json
from pathlib import Path
SCHEMA='public-controller-trace-v1'
DIRECTIONS={'NORTH','NORTHEAST','EAST','SOUTHEAST','SOUTH','SOUTHWEST','WEST','NORTHWEST','CENTER'}
# Overloads remain distinct: default attack/pickup is not an invented amount.
OPERATIONS={
 'moveForward':(), 'move':('direction',), 'turn':('direction',),
 'buildRat':('location',),'becomeRatKing':(), 'placeDirt':('location',),
 'removeDirt':('location',),'placeRatTrap':('location',),'removeRatTrap':('location',),
 'placeCatTrap':('location',),'removeCatTrap':('location',),'pickUpCheese':('location',),
 'pickUpCheeseAmount':('location','int'),'attack':('location',),'attackAmount':('location','int'),
 'squeak':('int',),'writeSharedArray':('int','int'),'transferCheese':('location','int'),
 'throwRat':(),'dropRat':('direction',),'carryRat':('location',),'disintegrate':(),'resign':(),
}
MAX_RECORD_BYTES=1024*1024

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def exact(obj,keys):
 if not isinstance(obj,dict) or set(obj)!=set(keys):raise ValueError('unknown or missing fields')
def integer(value,minimum=0):
 if type(value)!=int or value<minimum:raise ValueError('invalid integer')
def hash_value(value):
 if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('invalid hash')
def arguments(operation,args):
 if operation not in OPERATIONS or not isinstance(args,list) or len(args)!=len(OPERATIONS[operation]):raise ValueError('unknown operation/arity')
 for kind,value in zip(OPERATIONS[operation],args):
  if kind=='int':
   integer(value,-2**31)
   if value>=2**31:raise ValueError('integer overflow')
  elif kind=='direction':
   if not isinstance(value,str) or value not in DIRECTIONS:raise ValueError('invalid direction')
  else:
   if not isinstance(value,list) or len(value)!=2:raise ValueError('invalid location')
   for coordinate in value:integer(coordinate,-2**31)

def validate(record):
 exact(record,('schema','game_id','sequence','robot_id','team','round','turn','kind','payload','previous_sha256','sha256'))
 if record['schema']!=SCHEMA:raise ValueError('wrong schema')
 hash_value(record['game_id']);hash_value(record['previous_sha256']);hash_value(record['sha256'])
 for key in ('sequence','robot_id','round','turn'):integer(record[key])
 if record['team'] not in ('A','B'):raise ValueError('invalid player team')
 payload=record['payload'];kind=record['kind']
 if kind=='action':
  exact(payload,('operation','arguments','outcome','return_boolean','exception','public_observation_sha256','bytecodes_before','bytecodes_after'))
  arguments(payload['operation'],payload['arguments']);hash_value(payload['public_observation_sha256'])
  if payload['outcome'] not in ('returned','exception'):raise ValueError('unknown outcome')
  if payload['return_boolean'] is not None and type(payload['return_boolean'])!=bool:raise ValueError('bad boolean')
  if payload['operation']=='squeak' and payload['outcome']=='returned' and type(payload['return_boolean'])!=bool:raise ValueError('squeak result required')
  if payload['operation']!='squeak' and payload['return_boolean'] is not None:raise ValueError('unexpected return')
  if (payload['outcome']=='exception') != (isinstance(payload['exception'],str) and bool(payload['exception'])):raise ValueError('exception mismatch')
  if payload['outcome']=='returned' and payload['exception'] is not None:raise ValueError('unexpected exception')
  for key in ('bytecodes_before','bytecodes_after'):integer(payload[key])
 elif kind=='turn_end':
  exact(payload,('public_observation_sha256','physical_state_sha256','bytecodes_used','termination'))
  hash_value(payload['public_observation_sha256']);hash_value(payload['physical_state_sha256']);integer(payload['bytecodes_used'])
  if payload['termination'] not in ('yield','death','resign','overrun','exception'):raise ValueError('unknown turn termination')
 else:raise ValueError('unknown event kind')
 if digest({k:v for k,v in record.items() if k!='sha256'})!=record['sha256']:raise ValueError('record hash mismatch')
 if len(json.dumps(record).encode())>MAX_RECORD_BYTES:raise ValueError('record too large')
 return record

class TraceWriter:
 """Exclusive journal; crashes leave an incomplete file, never a completion seal."""
 def __init__(self,path,game_id):
  hash_value(game_id);self.game_id=game_id;self.sequence=0;self.previous='0'*64
  self.stream=Path(path).open('x');self.turns={};self.closed=False
 def append(self,robot_id,team,round,turn,kind,payload):
  if self.closed:raise ValueError('closed journal')
  key=(team,robot_id);prior=self.turns.get(key)
  if prior and (round,turn)<prior:raise ValueError('robot history moved backward')
  record={'schema':SCHEMA,'game_id':self.game_id,'sequence':self.sequence,'robot_id':robot_id,'team':team,'round':round,'turn':turn,'kind':kind,'payload':payload,'previous_sha256':self.previous}
  record['sha256']=digest(record);validate(record)
  self.stream.write(json.dumps(record,sort_keys=True,separators=(',',':'))+'\n');self.stream.flush()
  self.turns[key]=(round,turn);self.previous=record['sha256'];self.sequence+=1
  return record
 def close(self):self.stream.close();self.closed=True

def read_trace(path):
 records=[];previous='0'*64;game_id=None;turns={}
 with Path(path).open() as stream:
  while line:=stream.readline(MAX_RECORD_BYTES+1):
   if len(line.encode())>MAX_RECORD_BYTES or not line.endswith('\n'):raise ValueError('oversize or truncated journal')
   row=validate(json.loads(line))
   if row['sequence']!=len(records) or row['previous_sha256']!=previous:raise ValueError('broken journal chain')
   if game_id is not None and row['game_id']!=game_id:raise ValueError('mixed games')
   key=(row['team'],row['robot_id']);turn=(row['round'],row['turn'])
   if key in turns and turn<turns[key]:raise ValueError('robot history moved backward')
   turns[key]=turn;game_id=row['game_id'];previous=row['sha256'];records.append(row)
 return records

def compare_complete(reference,recorded,reference_seal,recorded_seal):
 """Full equality includes primitive arguments, exceptions, observations and budgets.

 Seals require full-schedule terminal proof. External engine integration must
 create them only after terminal completion; this function cannot prove origin.
 """
 for rows,seal in ((reference,reference_seal),(recorded,recorded_seal)):
  exact(seal,('game_id','complete','record_count','last_sha256','terminal_sha256','runtime_identity','teacher_identity','source_spec_sha256'))
  if seal['complete'] is not True or not rows:raise ValueError('incomplete trace')
  for key in ('game_id','last_sha256','terminal_sha256','runtime_identity','teacher_identity','source_spec_sha256'):hash_value(seal[key])
  if seal['record_count']!=len(rows) or seal['last_sha256']!=rows[-1]['sha256'] or seal['game_id']!=rows[0]['game_id']:raise ValueError('seal mismatch')
  previous='0'*64
  for i,row in enumerate(rows):
   validate(row)
   if row['sequence']!=i or row['previous_sha256']!=previous or row['game_id']!=seal['game_id']:raise ValueError('broken trace')
   previous=row['sha256']
  if rows[-1]['kind']!='turn_end':raise ValueError('trace ends inside a turn')
 if reference_seal!=recorded_seal or reference!=recorded:raise ValueError('instrumentation changes observable trace, budget, or terminal result')
 return {'equivalent':True,'records':len(reference),'training_approved':False,'scope':'saved trace equality only; engine capture provenance and dataset gates remain required'}
