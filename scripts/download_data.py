"""Download the public churn dataset."""

from pathlib import Path

from ai_ml_research_lab.data import download_dataset

if __name__ == "__main__":
    destination = Path(__file__).parents[1] / "data" / "raw" / "telco_churn.csv"
    download_dataset(destination)
    print(f"Downloaded source data to {destination}")
