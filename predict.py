"""Score a CSV of session records with the saved model."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=root/"artifacts/new_predictions.csv")
    args = parser.parse_args()
    # Load only this project's trusted artifact; joblib can execute serialized code.
    saved = joblib.load(root/"artifacts/model.joblib")
    rows = pd.read_csv(args.csv)
    with threadpool_limits(limits=2):
        probability = saved["pipeline"].predict_proba(rows)[:, 1]
    result = pd.DataFrame({"row_number": range(1, len(rows)+1), "purchase_score": probability,
        "predicted_purchase": probability >= saved["threshold"]})
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
