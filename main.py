"""End-to-end: generate cohort -> SQL analytics + dashboard -> train early-alert model."""
import argparse, logging, os
from generate_data import generate
import analytics, model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=4000)
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.makedirs("data", exist_ok=True)
    generate(n=args.students).to_csv("data/students.csv", index=False)
    analytics.run()
    model.run(tune=not args.no_tune)

if __name__ == "__main__":
    main()
