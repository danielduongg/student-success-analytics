"""Score a single student's dropout risk with the trained model.
Example:
    python predict.py --hs-gpa 2.4 --first-term-gpa 1.8 --lms 2 --unmet-need 9000 --first-gen 1
"""
import argparse, joblib, pandas as pd
from model import NUM, CAT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--major", default="Business")
    ap.add_argument("--age", type=int, default=19)
    ap.add_argument("--first-gen", type=int, default=0)
    ap.add_argument("--hs-gpa", type=float, default=3.2)
    ap.add_argument("--credits", type=int, default=15)
    ap.add_argument("--lms", type=int, default=6)
    ap.add_argument("--unmet-need", type=int, default=5000)
    ap.add_argument("--distance", type=int, default=100)
    ap.add_argument("--first-term-gpa", type=float, default=2.8)
    a = ap.parse_args()
    model = joblib.load("reports/model.joblib")
    row = pd.DataFrame([{ "age": a.age, "hs_gpa": a.hs_gpa, "credits_attempted": a.credits,
        "weekly_lms_logins": a.lms, "unmet_need_usd": a.unmet_need,
        "distance_from_home_mi": a.distance, "first_term_gpa": a.first_term_gpa,
        "major": a.major, "first_gen": a.first_gen }])[NUM + CAT]
    risk = 1 - float(model.predict_proba(row)[0, 1])
    print(f"Predicted dropout risk: {risk:.1%}")

if __name__ == "__main__":
    main()
