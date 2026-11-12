import pytest
from alpha0.imitation.runtime import runtime_spec, validate_runtime, launch_command, clean_environment, INJECTION

FLAGS=['--add-opens=java.base/java.lang=ALL-UNNAMED']
def setup(tmp_path):
    for name in ('bin/java','lib/modules','lib/server/libjvm.so','conf/security/java.security','release'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    return runtime_spec(tmp_path,FLAGS)

@pytest.mark.parametrize('name',['bin/java','lib/modules','lib/server/libjvm.so','conf/security/java.security','release','lib/new.so'])
def test_runtime_mutation(tmp_path,name):
    spec=setup(tmp_path);(tmp_path/name).write_text('changed')
    with pytest.raises(ValueError,match='identity'):validate_runtime(spec,tmp_path,FLAGS)

def test_launch_uses_pins(tmp_path):
    spec=setup(tmp_path)
    assert launch_command(spec,classpath='a:b',arguments=['x'])==[str(tmp_path/'bin/java'),*FLAGS,'-cp','a:b','WorldJournalTest','x']
    spec['flags']=[]
    with pytest.raises(ValueError,match='identity'):launch_command(spec,classpath='a:b',arguments=[])
    with pytest.raises(ValueError):runtime_spec(tmp_path,['-javaagent:bad.jar'])

@pytest.mark.parametrize('key',INJECTION)
def test_injection_rejected(key):
    with pytest.raises(ValueError,match='injection'):clean_environment({key:'injected'})
    assert clean_environment({key:''})=={key:''}
