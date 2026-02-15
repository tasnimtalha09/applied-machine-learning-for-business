# importing necessary libraries
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# importing the predict_leads function from the lead_evaluator module
from lead_evaluator import predict_leads


# parsing command-line arguments for the scoring script
def parse_args() -> argparse.Namespace:

    # creating an argument parser with a description
    parser = argparse.ArgumentParser(
        description = "Score Swan Chemical leads using the saved Lead Evaluator artifacts."
    )

    # adding a positional argument for the input CSV file path
    parser.add_argument(
        "input_csv",
        type = str,
        help = "Path to the input CSV file containing new leads (can include actual lead_bucket/lead_score).",
    )

    # adding an optional argument to save the scored output as CSV
    parser.add_argument(
        "--out",
        type = str,
        default = None,
        help = "Optional path to save the scored output as CSV. If omitted, nothing is saved.",
    )

    # parsing and returning the arguments
    return parser.parse_args()


# creating the main function to load leads, score them, and display/save results
def main() -> None:

    # parsing command-line arguments
    args = parse_args()

    # resolving the input CSV path and checking if it exists
    input_path = Path(args.input_csv).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # loading the new leads from the CSV file
    new_leads = pd.read_csv(filepath_or_buffer = input_path)

    # if the file contains a "lead_score" column, rename it to preserve for comparison
    if "lead_score" in new_leads.columns:
        new_leads = new_leads.rename(columns = {"lead_score": "actual_lead_score"})

    # predicting the lead buckets, scores, and actions using the lead_evaluator module
    results = predict_leads(new_leads)

    # building the output table starting with company names
    out = pd.DataFrame({"Company Name": results["company_name"]})

    # including actual bucket if it exists in the results
    if "lead_bucket" in results.columns:
        out["Actual Lead Bucket"] = results["lead_bucket"]

    # adding the predicted lead bucket
    out["Predicted Lead Bucket"] = results["pred_bucket"]

    # including actual lead score if it exists in the results
    if "actual_lead_score" in results.columns:
        out["Actual Lead Score"] = results["actual_lead_score"]

    # adding the predicted lead score
    out["Predicted Lead Score"] = results["lead_score"]

    # calculating score difference only if actual scores exist
    if "Actual Lead Score" in out.columns:
        out["Score Difference"] = (out["Actual Lead Score"] - out["Predicted Lead Score"]).abs().round(1)

    # adding the recommended action
    out["Action"] = results["action"]

    # printing the results table to the console
    print(out.to_string(index=False))

    # computing and printing bucket accuracy if actual labels exist and there are multiple rows
    if len(out) > 1 and "Actual Lead Bucket" in out.columns:
        matches = (out["Actual Lead Bucket"].values == out["Predicted Lead Bucket"].values)
        correct = int(matches.sum())
        total = int(len(out))
        acc = 100.0 * correct / total
        print(f"\nBucket Accuracy: {acc:.0f}% ({correct}/{total})")

    # computing and printing average score difference if actual scores exist
    if len(out) > 1 and "Score Difference" in out.columns:
        avg_diff = float(out["Score Difference"].mean())
        print(f"Average Score Difference: {avg_diff:.1f} points")

    # saving the scored output to a CSV file if the --out argument is provided
    if args.out is not None:
        out_path = Path(args.out).expanduser().resolve()
        out.to_csv(out_path, index=False)
        print(f"\nSaved scored output to: {out_path}")


# running the main function when the script is executed directly
if __name__ == "__main__":
    main()