import pandas as pd
import glob
from scipy.stats import mannwhitneyu

def parse_metrics(filepath):
    """
    Parse a results file in the given format.
    Expected lines include:
      MAE: [value]
      MMRE: [value]
      PRED: value
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    metrics = {"MAE": [], "MMRE": [], "PRED50": []}
    for line in lines:
        line = line.strip()
        if line.startswith("MAE:"):
            try:
                value = float(line.split('[')[1].split(']')[0])
                metrics["MAE"].append(value)
            except Exception as e:
                print(f"Error parsing MAE in {filepath}: {e}")
        elif line.startswith("MMRE:"):
            try:
                value = float(line.split('[')[1].split(']')[0])
                metrics["MMRE"].append(value)
            except Exception as e:
                print(f"Error parsing MMRE in {filepath}: {e}")
        elif line.startswith("PRED:"):
            try:
                value = float(line.split(':')[1].strip())
                metrics["PRED50"].append(value)
            except Exception as e:
                print(f"Error parsing PRED in {filepath}: {e}")
    return metrics

def compare_two_groups(replication_file, modified_file, project_name):
    """
    Compare two sets of results for a given project using Mann-Whitney U test for all metrics.
    """
    replication = parse_metrics(replication_file)
    modified = parse_metrics(modified_file)

    # Mann-Whitney U tests for all metrics
    mae_stat, mae_p = mannwhitneyu(replication["MAE"], modified["MAE"], alternative='two-sided')
    mmre_stat, mmre_p = mannwhitneyu(replication["MMRE"], modified["MMRE"], alternative='two-sided')
    pred_stat, pred_p = mannwhitneyu(replication["PRED50"], modified["PRED50"], alternative='two-sided')

    result_df = pd.DataFrame({
        "Project": [project_name] * 3,
        "Metric": ["MAE", "MMRE", "PRED(50)"],
        "Test": ["Mann-Whitney U"] * 3,
        "Statistic": [mae_stat, mmre_stat, pred_stat],
        "p-value": [mae_p, mmre_p, pred_p],
        "Significant": ["Yes" if p < 0.05 else "No" for p in [mae_p, mmre_p, pred_p]]
    })

    return result_df

# Statistical Significance Between Experiment 1 and Experiment 2 Results for Each Open-Source Project
result_mesos = compare_two_groups("EXPERIMENT 1 - Replication of Original Fines-SE Results/result_bert/mesos.txt",
                                "EXPERIMENT 2 - Replication using Redo-Fines-SE Dataset/results/MESOS/MESOS.txt",
                                "MESOS")
print("\nMESOS Results:")
print(result_mesos)

result_usergrid = compare_two_groups("EXPERIMENT 1 - Replication of Original Fines-SE Results/result_bert/usergrid.txt",
                                   "EXPERIMENT 2 - Replication using Redo-Fines-SE Dataset/results/USERGRID/USERGRID.txt",
                                   "USERGRID")
print("\nUSERGRID Results:")
print(result_usergrid)

result_datamanagement = compare_two_groups("EXPERIMENT 1 - Replication of Original Fines-SE Results/result_bert/datamanagement.txt",
                                         "EXPERIMENT 2 - Replication using Redo-Fines-SE Dataset/results/DATA_MANAGEMENT/DATA_MANAGEMENT.txt",
                                         "DATA_MANAGEMENT")
print("\nDATA_MANAGEMENT Results:")
print(result_datamanagement)

# Optionally, combine all results into a single DataFrame
all_results = pd.concat([result_mesos, result_usergrid, result_datamanagement], ignore_index=True)
print("\nAll Results:")
print(all_results)
