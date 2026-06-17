"""Generate a realistic first-year student cohort for retention modeling.

Synthetic, but the retention target is driven by the factors that actually
predict college retention in the literature (prior academic performance,
engagement, financial need, first-gen status, course load). This keeps the
repo reproducible while behaving like real institutional data. See README
for swapping in the public UCI 'Student Performance' dataset.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

MAJORS = ["Business", "Nursing", "Computer Science", "Psychology",
          "Education", "Biology", "Engineering", "Communications"]

def generate(n: int = 4000, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hs_gpa = np.clip(rng.normal(3.3, 0.45, n), 1.8, 4.0)
    first_gen = rng.binomial(1, 0.34, n)
    # engagement: weekly LMS logins (counts) -> Poisson, higher for stronger students
    lms_logins = rng.poisson(np.clip(6 + (hs_gpa - 3.3) * 4, 1, None))
    credits = rng.choice([12, 13, 15, 16, 18], n, p=[.18, .12, .4, .2, .1])
    unmet_need = np.clip(rng.normal(6000, 4000, n), 0, 20000)  # $ financial gap
    distance_mi = np.abs(rng.normal(120, 200, n))
    age = np.clip(rng.normal(19.5, 3.0, n), 17, 45).round(0)
    major = rng.choice(MAJORS, n)
    cohort = rng.choice([2021, 2022, 2023], n)
    # first-term GPA: correlated with hs_gpa + engagement + noise
    term_gpa = np.clip(0.55 * hs_gpa + 0.18 * (lms_logins / 8)
                       + rng.normal(0.7, 0.45, n), 0.0, 4.0)

    # ---- latent retention propensity (log-odds of being RETAINED) ----
    z = (1.9
         + 1.7 * (term_gpa - 2.5)
         + 0.9 * (hs_gpa - 3.3)
         + 0.08 * (lms_logins - 6)
         - 0.45 * first_gen
         - 0.00006 * unmet_need
         - 0.04 * (credits - 15) ** 2 / 3     # over/under-loading hurts
         + rng.normal(0, 0.6, n))
    p_ret = 1 / (1 + np.exp(-z))
    retained = rng.binomial(1, p_ret)

    return pd.DataFrame(dict(
        student_id=np.arange(100000, 100000 + n),
        entry_cohort=cohort, major=major, age=age.astype(int),
        first_gen=first_gen, hs_gpa=hs_gpa.round(2),
        credits_attempted=credits, weekly_lms_logins=lms_logins,
        unmet_need_usd=unmet_need.round(0).astype(int),
        distance_from_home_mi=distance_mi.round(0).astype(int),
        first_term_gpa=term_gpa.round(2),
        retained_to_year2=retained,
    ))

if __name__ == "__main__":
    df = generate()
    df.to_csv("data/students.csv", index=False)
    print(f"Wrote {len(df):,} students | retention rate {df.retained_to_year2.mean():.3f}")
