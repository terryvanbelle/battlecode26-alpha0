from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from alpha0 import constraint_v7_report as R
from scripts.plot_constraint_v7 import generate


class ConstraintV7ReportTest(unittest.TestCase):
    def setUp(self):
        self.targets=[1]*9+[0]*23; self.baseline=[0]*32; self.masks=[[True]*40 for _ in range(32)]
        self.proof={**{key:True for key in R.PROOF_FLAGS},"updates":128}

    def snapshots(self,one,zero):
        return {arm:{str(step):{"root_count":32,"root_selected_action_matches":count}
                     for step,count in zip(R.STEPS,counts)} for arm,counts in (("kl_1",one),("kl_0",zero))}

    def decision(self,rows,**changes):
        return R.classify(rows,self.proof,**{"fit_complete":True,"integrity_verified":True,**changes})

    def test_ordered_budget_even_after_regression_and_anchoring_and_neither(self):
        result=self.decision(self.snapshots([23,32,31],[32,32,32]))
        self.assertEqual(result["classification"],"budget"); self.assertTrue(result["regressed_after_full_match"]["kl_1"])
        self.assertFalse(result["anchoring_evidence"])
        self.assertEqual(self.decision(self.snapshots([23,27,31],[31,32,32]))["classification"],"anchoring")
        self.assertEqual(self.decision(self.snapshots([23,27,31],[23,30,31]))["classification"],"neither")

    def test_missing_snapshot_failed_reproduction_and_integrity_never_classify(self):
        rows=self.snapshots([23,32,32],[32,32,32]); del rows["kl_0"]["512"]
        self.assertIsNone(self.decision(rows)["classification"])
        rows=self.snapshots([23,32,32],[32,32,32])
        self.assertIsNone(self.decision(rows,integrity_verified=False)["classification"])
        self.proof["optimizer_state_equal"]=False
        self.assertIsNone(self.decision(rows)["classification"])
        with self.assertRaises(ValueError):
            R.classify(self.snapshots([32,32,32],[32,32,32]),{**self.proof,"optimizer_state_equal":True},fit_complete=True,integrity_verified=True)

    def metrics(self):
        return {"root_actions":self.baseline,"root_count":32,"root_selected_action_matches":23,"root_target_agreement":23/32,
            "discordant_target_count":9,"discordant_target_matches":0,"retained_target_count":23,"retained_target_matches":23,
            "root_action_changed_fraction":0.,"anchor_action_changes":3,"anchor_action_changed_fraction":3/512,
            "target_ce":.9,"anchor_kl":.05,"loss":.95,"expected_anchor_kl":.06,"expected_objective":.96,
            "parameter_displacement_l2":.2,"target_ce_gradient_norm":.3,"balanced_anchor_kl_gradient_norm":.4,
            "expected_total_gradient_norm":.2,"ce_kl_gradient_cosine":-.8}

    def test_agreement_is_recomputed_from_all_actions_and_weighted_objective(self):
        result=R.validate_metrics(self.metrics(),self.targets,self.baseline,self.masks,1)
        self.assertEqual(result["weighted_anchor_kl_contribution"],.06)
        for key,value in (("discordant_target_matches",9),("root_actions",[0]*31),("expected_objective",.95)):
            metric=self.metrics(); metric[key]=value
            with self.assertRaises(ValueError): R.validate_metrics(metric,self.targets,self.baseline,self.masks,1)

    def test_trace_uses_recorded_updates_and_correct_kl_coefficient(self):
        history=[{"step":1,"target_ce":.8,"anchor_kl":.2,"loss":.8}]
        self.assertEqual(R.validate_history(history,1,0)[0]["weighted_anchor_kl"],0)
        with self.assertRaises(ValueError): R.validate_history(history,1,1)
        with self.assertRaises(ValueError): R.validate_history(history,2,0)

    def test_empty_report_and_plots_have_no_classification_or_invented_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); dataset=root/"training.npz"
            prior=np.zeros((32,40),np.float32); prior[:,0]=1
            np.savez(dataset,terminal_actions=np.array(self.targets),root_prior=prior,root_mask=np.ones((32,40),bool))
            config={"experiment_id":"cached-head-constraint-v7","updates":2048,"snapshots":[128,512,2048],
                    "training_roots":32,"anchor_count":512,"dataset":{"path":str(dataset),"sha256":R.C.sha256(dataset)}}
            identity={"config":config,"inputs":{},"sources":{}}
            result=R.report(root/"run",root/"progress",config,identity,integrity_verified=True)
            self.assertFalse(result["complete"]); self.assertIsNone(result["decision"]["classification"])
            self.assertEqual(result["arms"]["kl_1"]["history"],[])
            generate(result,root/"plots")
            self.assertTrue((root/"plots/learning-curves.png").is_file())
            self.assertIn("Pending",(root/"plots/README.md").read_text())


if __name__=="__main__": unittest.main()
