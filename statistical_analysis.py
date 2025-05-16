import pandas as pd
import glob
from scipy.stats import mannwhitneyu, chi2_contingency


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
    Compare two sets of results for a given project.
    Uses Mann-Whitney U for continuous metrics and Chi-square for PRED(50).
    """
    replication = parse_metrics(replication_file)
    modified = parse_metrics(modified_file)

    # Mann-Whitney U tests for continuous metrics:
    mae_stat, mae_p = mannwhitneyu(replication["MAE"], modified["MAE"], alternative='two-sided')
    mmre_stat, mmre_p = mannwhitneyu(replication["MMRE"], modified["MMRE"], alternative='two-sided')

    # Chi-square test for categorical metric (PRED(50)):
    rep_success = [1 if x >= 0.5 else 0 for x in replication["PRED50"]]
    mod_success = [1 if x >= 0.5 else 0 for x in modified["PRED50"]]
    # Build a contingency table:
    contingency_table = pd.crosstab(
        ["Replication"] * len(rep_success) + ["Modified"] * len(mod_success),
        rep_success + mod_success
    )
    chi2_stat, chi2_p, _, _ = chi2_contingency(contingency_table)

    result_df = pd.DataFrame({
        "Project": [project_name] * 3,
        "Metric": ["MAE", "MMRE", "PRED(50)"],
        "Test": ["Mann-Whitney U", "Mann-Whitney U", "Chi-square"],
        "Statistic": [mae_stat, mmre_stat, chi2_stat],
        "p-value": [mae_p, mmre_p, chi2_p],
        "Significant": ["Yes" if p < 0.05 else "No" for p in [mae_p, mmre_p, chi2_p]]
    })

    return result_df


# Statistical Significance Between Experiment 1 and Experiment 2 Results for Each Open-Source Project
result_mesos = compare_two_groups("FineSE/result_bert/mesos.txt",
                                  "My-SE/results/MESOS/MESOS.txt",
                                  "MESOS")
print(result_mesos)

result_usergrid = compare_two_groups("FineSE/result_bert/usergrid.txt",
                                     "My-SE/results/USERGRID/USERGRID.txt",
                                     "USERGRID")
print(result_usergrid)

result_datamanagement = compare_two_groups("FineSE/result_bert/datamanagement.txt",
                                           "My-SE/results/DATA_MANAGEMENT/DATA_MANAGEMENT.txt",
                                           "DATA_MANAGEMENT")
print(result_datamanagement)


# Statistical Significance of Experiment 3: Comparing Non-Hyperparameter and Hyperparameter Configurations Across
# Open-Source Projects

# codeBERT
# exp_3_codebert_result_mesos = compare_two_groups("My-SE-v2/results/codeBERT/default/MESOS/MESOS.txt",
#                                                 "My-SE-v2/results/codeBERT/hyper_param/MESOS/MESOS.txt",
#                                                 "MESOS")
# print(exp_3_codebert_result_mesos)

# exp_3_codebert_result_usergrid = compare_two_groups("My-SE-v2/results/codeBERT/default/USERGRID/USERGRID.txt",
#                                                    "My-SE-v2/results/codeBERT/hyper_param/USERGRID/USERGRID.txt",
#                                                    "USERGRID")
# print(exp_3_codebert_result_usergrid)

# exp_3_codebert_result_dm = compare_two_groups("My-SE-v2/results/codeBERT/default/DATA_MANAGEMENT/DATA_MANAGEMENT.txt",
#                                              "My-SE-v2/results/codeBERT/hyper_param/DATA_MANAGEMENT"
#                                              "/DATA_MANAGEMENT.txt",
#                                              "DATA_MANAGEMENT")
# print(exp_3_codebert_result_dm)


# BERT-codeBERT
#exp_3_bert_codebert_result_mesos = compare_two_groups("My-SE-v2/results/bert_codebert/default/MESOS/MESOS.txt",
#                                                 "My-SE-v2/results/bert_codebert/hyper_param/MESOS/MESOS.txt",
#                                                 "MESOS")
#print(exp_3_bert_codebert_result_mesos)

#exp_3_bert_codebert_result_usergrid = compare_two_groups("My-SE-v2/results/bert_codebert/default/USERGRID/USERGRID.txt",
#                                                    "My-SE-v2/results/bert_codebert/hyper_param/USERGRID/USERGRID.txt",
#                                                    "USERGRID")
#print(exp_3_bert_codebert_result_usergrid)

#exp_3_bert_codebert_result_dm = compare_two_groups("My-SE-v2/results/bert_codebert/default/DATA_MANAGEMENT/DATA_MANAGEMENT.txt",
#                                              "My-SE-v2/results/bert_codebert/hyper_param/DATA_MANAGEMENT"
#                                              "/DATA_MANAGEMENT.txt",
#                                              "DATA_MANAGEMENT")
#print(exp_3_bert_codebert_result_dm)

# BERT-codeBERT-autoencoders
#exp_3_autoencoders_result_mesos = compare_two_groups("My-SE-v2/results/autoencoders/default/MESOS/MESOS.txt",
#                                                 "My-SE-v2/results/autoencoders/hyper_param/MESOS/MESOS.txt",
#                                                 "MESOS")
#print(exp_3_autoencoders_result_mesos)

#exp_3_autoencoders_result_usergrid = compare_two_groups("My-SE-v2/results/autoencoders/default/USERGRID/USERGRID.txt",
#                                                    "My-SE-v2/results/autoencoders/hyper_param/USERGRID/USERGRID.txt",
#                                                    "USERGRID")
#print(exp_3_autoencoders_result_usergrid)

#exp_3_autoencoders_result_dm = compare_two_groups("My-SE-v2/results/autoencoders/default/DATA_MANAGEMENT/DATA_MANAGEMENT.txt",
#                                              "My-SE-v2/results/autoencoders/hyper_param/DATA_MANAGEMENT"
#                                              "/DATA_MANAGEMENT.txt",
#                                              "DATA_MANAGEMENT")
#print(exp_3_autoencoders_result_dm)
