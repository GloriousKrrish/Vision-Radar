import os
import sys
import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.abspath("packages"))
from visionradar.models.database import SessionLocal
from visionradar.models.entities import Experiment

def export_paper_tables(output_dir: str = "research"):
    """
    Exports experiment metrics from SQLite database into LaTeX and CSV tables for academic publication.
    """
    os.makedirs(output_dir, exist_ok=True)
    db: Session = SessionLocal()

    exps = db.query(Experiment).all()
    db.close()

    if not exps:
        print("No experiment records found in database.")
        return

    data = [
        {
            "Experiment": e.name,
            "Dataset": e.dataset_name,
            "Detector": e.detector_name,
            "Tracker": e.tracker_name,
            "Speed Method": e.speed_method,
            "MAE (km/h)": e.mae_kmh,
            "RMSE (km/h)": e.rmse_kmh,
            "R2 Score": e.r2_score
        }
        for e in exps
    ]

    df = pd.DataFrame(data)

    csv_path = os.path.join(output_dir, "experiment_results.csv")
    tex_path = os.path.join(output_dir, "table_benchmark_results.tex")

    df.to_csv(csv_path, index=False)

    # Format custom standalone LaTeX table
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Monocular Speed Estimation Benchmark Results across Datasets and Baselines}",
        "\\label{tab:speed_benchmark}",
        "\\begin{tabular}{l l l l l r r r}",
        "\\hline",
        "Experiment & Dataset & Detector & Tracker & Speed Method & MAE (km/h) & RMSE (km/h) & $R^2$ Score \\\\",
        "\\hline"
    ]

    for d in data:
        line = f"{d['Experiment']} & {d['Dataset']} & {d['Detector']} & {d['Tracker']} & {d['Speed Method']} & {d['MAE (km/h)']:.2f} & {d['RMSE (km/h)']:.2f} & {d['R2 Score']:.4f} \\\\"
        latex_lines.append(line)

    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])

    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"Exported CSV table to: {csv_path}")
    print(f"Exported LaTeX paper table to: {tex_path}")

if __name__ == "__main__":
    export_paper_tables()
