import os
from generate_data import generate
import model as M

def test_model_and_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data"); os.makedirs("reports", exist_ok=True)
    generate(n=2500, seed=4).to_csv("data/students.csv", index=False)
    s = M.run(tune=False)
    assert s["models"]["random_forest"]["roc_auc"] > 0.6
    assert "fairness_dropout_recall" in s
    assert len(s["intervention_budget"]) == 5
    assert os.path.exists("reports/at_risk_students.csv")
    assert os.path.exists("reports/model.joblib")
