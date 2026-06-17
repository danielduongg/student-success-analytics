"""Run the analytics SQL (in-memory DuckDB) and build a retention dashboard."""
from __future__ import annotations
import os, re
import duckdb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _run_named_queries(con):
    sql = open("analytics_queries.sql").read()
    blocks = [b.strip() for b in sql.split(";") if b.strip() and not b.strip().startswith("--\n")]
    results = []
    for b in blocks:
        m = re.search(r"--\s*(Q\d:[^\n]+)", b)
        label = m.group(1) if m else b[:40]
        stmt = "\n".join(l for l in b.splitlines() if not l.strip().startswith("--"))
        if stmt.strip():
            results.append((label, con.execute(stmt).df()))
    return results

def run():
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv("data/students.csv")
    con = duckdb.connect(":memory:")
    con.register("students", df)

    print(f"Cohort: {len(df):,} students | overall retention "
          f"{df.retained_to_year2.mean():.1%}\n")
    for label, res in _run_named_queries(con):
        print(f"--- {label} ---")
        print(res.to_string(index=False), "\n")

    # ---- Power BI-style dashboard (4 panels) ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("First-Year Retention Dashboard", fontsize=15, fontweight="bold")

    g = df.groupby(pd.cut(df.first_term_gpa, [0, 2, 2.5, 3, 3.5, 4]),
                   observed=True).retained_to_year2.mean()
    ax[0, 0].bar([str(i) for i in g.index], g.values * 100, color="#2b6cb0")
    ax[0, 0].set_title("Retention by first-term GPA"); ax[0, 0].set_ylabel("% retained")
    ax[0, 0].tick_params(axis="x", rotation=20)

    fg = df.groupby("first_gen").retained_to_year2.mean() * 100
    ax[0, 1].bar(["Continuing-gen", "First-gen"], fg.values, color=["#2f855a", "#c05621"])
    ax[0, 1].set_title("Retention by first-gen status"); ax[0, 1].set_ylabel("% retained")

    df["eng_q"] = pd.qcut(df.weekly_lms_logins, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    eq = df.groupby("eng_q", observed=True).retained_to_year2.mean() * 100
    ax[1, 0].plot(eq.index.astype(str), eq.values, "o-", color="#6b46c1", lw=2)
    ax[1, 0].set_title("Retention by engagement quartile"); ax[1, 0].set_ylabel("% retained")

    co = df.groupby("entry_cohort").retained_to_year2.mean() * 100
    ax[1, 1].bar(co.index.astype(str), co.values, color="#2c5282")
    ax[1, 1].set_title("Retention by entry cohort"); ax[1, 1].set_ylabel("% retained")
    ax[1, 1].set_ylim(0, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("reports/retention_dashboard.png", dpi=120); plt.close()
    print("Saved reports/retention_dashboard.png")
    con.close()

if __name__ == "__main__":
    run()
