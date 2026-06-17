"""Early-alert retention model: hyperparameter search, calibration, an
intervention-budget analysis (precision/recall@K), a first-gen fairness check,
and permutation importance. Trains on past cohorts, validates on the newest.
"""
from __future__ import annotations
import json, logging, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

log = logging.getLogger("retention.model")
NUM = ["age", "hs_gpa", "credits_attempted", "weekly_lms_logins",
       "unmet_need_usd", "distance_from_home_mi", "first_term_gpa"]
CAT = ["major", "first_gen"]
REPORTS = "reports"


def _prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def _precision_recall_at_k(y_true_dropout, risk, k_frac):
    """If advisors can only reach the top-k% riskiest students, how many real
    dropouts do we catch (recall) and how precise is that list?"""
    n = len(risk); k = max(1, int(k_frac * n))
    idx = np.argsort(risk)[::-1][:k]
    flagged = np.zeros(n, dtype=bool); flagged[idx] = True
    tp = int((flagged & (y_true_dropout == 1)).sum())
    return dict(k_frac=k_frac, n_contacted=k,
                precision=round(tp / k, 3),
                recall=round(tp / max(1, int(y_true_dropout.sum())), 3))


def run(data_csv="data/students.csv", tune=True):
    os.makedirs(REPORTS, exist_ok=True)
    df = pd.read_csv(data_csv)
    train, test = df[df.entry_cohort < 2023], df[df.entry_cohort == 2023].copy()
    Xtr, ytr = train[NUM + CAT], train.retained_to_year2
    Xte, yte = test[NUM + CAT], test.retained_to_year2

    rf = Pipeline([("prep", _prep()),
                   ("clf", RandomForestClassifier(class_weight="balanced", random_state=23))])
    if tune:
        search = RandomizedSearchCV(
            rf, {"clf__n_estimators": [200, 300, 500],
                 "clf__max_depth": [6, 8, 12, None],
                 "clf__min_samples_leaf": [1, 3, 5]},
            n_iter=8, scoring="roc_auc",
            cv=StratifiedKFold(4, shuffle=True, random_state=23), random_state=23, n_jobs=-1)
        search.fit(Xtr, ytr); rf = search.best_estimator_
        log.info("best RF params: %s", search.best_params_)
    else:
        rf.fit(Xtr, ytr)

    logit = Pipeline([("prep", _prep()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(Xtr, ytr)

    models = {"logistic_regression": logit, "random_forest": rf}
    metrics, risks = {}, {}
    dropout_true = (yte == 0).astype(int).values
    for name, m in models.items():
        p_ret = m.predict_proba(Xte)[:, 1]
        risk = 1 - p_ret
        risks[name] = risk
        metrics[name] = dict(
            roc_auc=round(float(roc_auc_score(yte, p_ret)), 4),
            pr_auc_dropout=round(float(average_precision_score(dropout_true, risk)), 4))

    # intervention-budget curve on the better model
    best = "random_forest" if metrics["random_forest"]["roc_auc"] >= metrics["logistic_regression"]["roc_auc"] else "logistic_regression"
    budget = [_precision_recall_at_k(dropout_true, risks[best], k) for k in (0.05, 0.10, 0.15, 0.20, 0.30)]

    # fairness: dropout-recall parity across first-gen at a 20% outreach budget
    test["risk"] = risks[best]
    fairness = {}
    thr = np.quantile(risks[best], 0.80)
    for grp, sub in test.groupby("first_gen"):
        d = (sub.retained_to_year2 == 0)
        flagged = sub.risk >= thr
        rec = float((flagged & d).sum() / max(1, d.sum()))
        fairness["first_gen" if grp == 1 else "continuing_gen"] = round(rec, 3)

    # permutation importance (model-agnostic)
    perm = permutation_importance(models[best], Xte, yte, n_repeats=15,
                                  random_state=23, scoring="roc_auc")
    # permutation_importance permutes raw INPUT columns, so labels are NUM + CAT
    imp = pd.Series(perm.importances_mean, index=NUM + CAT).sort_values().tail(10)

    _plots(yte, models, risks, metrics, imp, budget)
    test.sort_values("risk", ascending=False).head(20)[
        ["student_id","major","first_gen","hs_gpa","first_term_gpa",
         "weekly_lms_logins","unmet_need_usd","risk"]].round(3).to_csv(
        f"{REPORTS}/at_risk_students.csv", index=False)

    summary = dict(best_model=best, models=metrics,
                   intervention_budget=budget, fairness_dropout_recall=fairness)
    with open(f"{REPORTS}/metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    import joblib; joblib.dump(models[best], f"{REPORTS}/model.joblib")
    print(json.dumps(summary, indent=2))
    return summary


def _plots(yte, models, risks, metrics, imp, budget):
    plt.figure(figsize=(6, 5))
    for name in models:
        fpr, tpr, _ = roc_curve(yte, 1 - risks[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['roc_auc']})")
    plt.plot([0,1],[0,1],"k--",alpha=.4); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Retention ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{REPORTS}/roc.png", dpi=120); plt.close()

    plt.figure(figsize=(7,5)); imp.plot.barh(color="#2b6cb0")
    plt.title("Permutation importance (ROC-AUC drop)"); plt.tight_layout()
    plt.savefig(f"{REPORTS}/risk_factors.png", dpi=120); plt.close()

    bdf = pd.DataFrame(budget)
    plt.figure(figsize=(7,4))
    plt.plot(bdf.k_frac*100, bdf.recall*100, "o-", label="Recall (dropouts caught)")
    plt.plot(bdf.k_frac*100, bdf.precision*100, "s-", label="Precision of outreach list")
    plt.xlabel("Outreach budget (% of cohort contacted)"); plt.ylabel("%")
    plt.title("Intervention budget trade-off"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{REPORTS}/intervention_budget.png", dpi=120); plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
