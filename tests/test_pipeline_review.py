"""Independent end-to-end checks for checkpoint continuity and engine scheduling."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from alpha0.network import NetworkConfig, PolicyValueNet, policy_value_loss
from alpha0.train import load_model, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_restores_identical_next_optimizer_update(tmp_path):
    """A real AdamW step before/after serialization detects missing optimizer/RNG state."""
    torch.set_num_threads(1)
    torch.manual_seed(43)
    rng = np.random.default_rng(91)
    net = PolicyValueNet(NetworkConfig(3, 2, 4, channels=8, blocks=1))
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    board, scalars = torch.randn(3, 3, 9, 9), torch.randn(3, 2)
    policy = torch.tensor([[0.5, 0.25, 0.25, 0.0]] * 3)
    outcome = torch.tensor([1.0, -1.0, 1.0])

    def update(model, opt):
        logits, values = model(board, scalars)
        loss, _ = policy_value_loss(logits, values, policy, outcome)
        opt.zero_grad()
        loss.backward()
        opt.step()

    update(net, optimizer)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, net, optimizer, 3, {"seed": 91}, rng, 6, 500, 12.0)
    expected_numpy = rng.integers(0, 2**32, size=12)
    expected_torch = torch.rand(12)
    update(net, optimizer)

    restored, payload = load_model(checkpoint)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
    restored_optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng.bit_generator.state = payload["numpy_rng"]
    torch.set_rng_state(payload["torch_rng"])
    assert np.array_equal(expected_numpy, rng.integers(0, 2**32, size=12))
    assert torch.equal(expected_torch, torch.rand(12))
    update(restored, restored_optimizer)
    for expected, actual in zip(net.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


NATIVE_EQUIVALENCE_SOURCE = r'''
package alpha0;
import battlecode.common.*;
import battlecode.crossplay.CrossPlayLanguage;
import battlecode.server.*;
import battlecode.world.*;
import battlecode.world.control.*;
import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.lang.reflect.*;
import java.util.*;
import sun.misc.Unsafe;

public class NativeStepEquivalence {
    static int choose(EngineBridge.Session s, Random rng, List<Integer> actions) {
        List<Integer> legal=s.legalActions();
        if(legal.size()==1) return 0;
        int action=legal.get(rng.nextInt(legal.size()));
        actions.add(action);
        return action;
    }
    static EngineBridge.Session emptySession() throws Exception {
        Field field=Unsafe.class.getDeclaredField("theUnsafe");field.setAccessible(true);
        return (EngineBridge.Session)((Unsafe)field.get(null)).allocateInstance(EngineBridge.Session.class);
    }
    public static void main(String[] args) throws Exception {
        ObjectMapper json=new ObjectMapper();
        String mapName=args[0]; int seed=Integer.parseInt(args[1]);
        JsonNode request=json.readTree("{\"map\":\""+mapName+"\",\"seed\":"+seed+",\"maps_dir\":\"maps\"}");
        EngineBridge.Session manual=new EngineBridge.Session(request);
        Random manualRandom=new Random(711);
        List<Integer> manualActions=new ArrayList<>();
        while(manual.world.isRunning()) manual.step(choose(manual,manualRandom,manualActions));
        Map<String,Object> expected=manual.state();

        EngineBridge.Session nativeSession=emptySession();
        Random nativeRandom=new Random(711);
        List<Integer> nativeActions=new ArrayList<>();
        RobotControlProvider learned=new NullControlProvider() {
            @Override public void matchStarted(GameWorld world) { nativeSession.world=world; }
            @Override public void runRobot(InternalRobot robot) {
                nativeSession.current=robot;
                int action=choose(nativeSession,nativeRandom,nativeActions);
                nativeSession.macro(action);
                nativeSession.current=null;
            }
        };
        TeamControlProvider providers=new TeamControlProvider();
        providers.registerControlProvider(Team.A,learned);
        providers.registerControlProvider(Team.B,learned);
        providers.registerControlProvider(Team.NEUTRAL,new NullControlProvider());
        LiveMap map=GameMapIO.loadMap(mapName,new File("maps"),false);
        Field seedField=LiveMap.class.getDeclaredField("seed");seedField.setAccessible(true);seedField.setInt(map,seed);
        InternalRobot.rand=new Random(1092);
        GameInfo info=new GameInfo("alpha0",CrossPlayLanguage.JAVA,"alpha0",null,"opponent",CrossPlayLanguage.JAVA,"opponent",null,new String[]{mapName},null,false);
        GameWorld world=new GameWorld(map,providers,new NoopGameMaker(info).getMatchMaker());
        while(world.isRunning()) world.runRound();
        nativeSession.decisions=nativeActions.size();
        Map<String,Object> actual=nativeSession.state();
        if(!manualActions.equals(nativeActions)) throw new AssertionError("Action trajectory differs: "+manualActions.size()+" vs "+nativeActions.size());
        JsonNode expectedJson=json.valueToTree(expected),actualJson=json.valueToTree(actual);
        if(!expectedJson.equals(actualJson)) {
            for(String key:expected.keySet()) if(!expectedJson.get(key).equals(actualJson.get(key))) System.err.println("DIFF "+key+": "+expectedJson.get(key)+" vs "+actualJson.get(key));
            throw new AssertionError("Final engine state differs");
        }
        System.out.println("Native runRound equivalence passed: "+mapName+", "+world.getCurrentRound()+" rounds, "+nativeActions.size()+" decisions");
    }
}
'''


@pytest.mark.skipif(not (ROOT / "build/java/alpha0/EngineBridge.class").exists(), reason="run scripts/setup_engine.py first")
@pytest.mark.parametrize("map_name,seed", [("tiny", 19), ("DefaultSmall", 781)])
def test_manual_stepper_matches_native_official_round_loop(tmp_path, map_name, seed):
    """Compare full games and observations using the unmodified runRound scheduler."""
    source = tmp_path / "NativeStepEquivalence.java"
    source.write_text(NATIVE_EQUIVALENCE_SOURCE)
    javac = ROOT / "vendor/jdk/bin/javac"
    java = ROOT / "vendor/jdk/bin/java"
    classpath = f"{ROOT / 'build/java'}:{ROOT / 'vendor/engine/battlecode26-java-1.2.6.jar'}"
    subprocess.run([str(javac) if javac.exists() else "javac", "-cp", classpath,
                    "-d", str(tmp_path), str(source)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(java) if java.exists() else "java", "-cp", f"{tmp_path}:{classpath}",
                             "alpha0.NativeStepEquivalence", map_name, str(seed)],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Native runRound equivalence passed" in result.stdout
