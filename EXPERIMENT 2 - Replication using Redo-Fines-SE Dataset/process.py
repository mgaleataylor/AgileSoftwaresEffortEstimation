import ast
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np


def clean_issue_data(df_issues, output_dir, project_name, issue_completion_key):
    df_clean = df_issues.copy()
    df_clean = df_clean[df_clean['Issue Key'].str.startswith(f'{project_name}-')]
    df_clean = df_clean[df_clean['Status'].isin(issue_completion_key)]

    df_clean['Story Points'] = pd.to_numeric(df_clean['Story Points'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Story Points'])

    df_clean["Story Points"] = df_clean["Story Points"].round()
    df_clean = df_clean[df_clean['Story Points'] > 0]

    # Remove timezone information from datetimes
    df_clean['Created'] = pd.to_datetime(df_clean['Created'], errors='coerce').dt.tz_localize(None)
    df_clean = df_clean.dropna(subset=['Created'])
    df_clean['Resolved'] = pd.to_datetime(df_clean['Resolved'], errors='coerce').dt.tz_localize(None)
    df_clean = df_clean.dropna(subset=['Resolved'])

    # Create a feature "Contributors" that counts distinct users among Assignee, Reporter, and Creator
    df_clean["Contributors"] = df_clean.apply(
        lambda row: len(pd.unique([
            x for x in row[["Assignee_User", "Reporter_User", "Creator_User"]]
            if pd.notnull(x) and str(x).strip() and str(x).strip() != "Unassigned"
        ])),
        axis=1
    )

    save_path = os.path.join(output_dir, "issues_cleaned.xlsx")
    df_clean.to_excel(save_path, index=False)
    print(f"Cleaned issues saved to: {save_path}")
    return df_clean


# ==========================================================
# 5. Aggregate Developer Features from PR & Issue Data
# ==========================================================
def aggregate_developer_features(df_pr, df_issues_clean, df_contrib, status_completion_keys):
    # Developer commits (sum of commits per PR creator)
    dev_commits = df_pr.groupby("PR User")["Commits"].sum().reset_index()
    dev_commits.rename(columns={"Commits": "Developer_commits", "PR User": "Developer"}, inplace=True)

    # Developer created MRs (count PRs per creator)
    dev_created_mrs = df_pr.groupby("PR User").size().reset_index(name="Developer_created_MRs")
    dev_created_mrs.rename(columns={"PR User": "Developer"}, inplace=True)

    # Developer modified files (sum of files modified per creator)
    dev_modified_files = df_pr.groupby("PR User")["Files Modified"].sum().reset_index()
    dev_modified_files.rename(columns={"Files Modified": "Developer_modified_files", "PR User": "Developer"},
                              inplace=True)

    # Developer commits reviews (count review events per reviewer)
    review_counts = {}
    for idx, row in df_pr.iterrows():
        reviews_data = row["Reviewers Info"]
        if isinstance(reviews_data, str):
            try:
                reviews = ast.literal_eval(reviews_data)
            except Exception:
                reviews = []
        elif isinstance(reviews_data, list):
            reviews = reviews_data
        else:
            reviews = []

        for review in reviews:
            reviewer = review.get("Reviewer")
            if (df_contrib['Login'] == reviewer).any():
                reviewer = df_contrib.loc[df_contrib['Login'] == reviewer, 'Name'].iloc[0]
            if reviewer:
                review_counts[reviewer] = review_counts.get(reviewer, 0) + 1
    dev_review_df = pd.DataFrame(list(review_counts.items()), columns=["Developer", "Developer_commits_reviews"])

    # Developer updated MRs (for PRs with > 1 commit, count unique developers from later commits)
    developer_updated_prs = {}
    for idx, row in df_pr.iterrows():
        try:
            commits_list = ast.literal_eval(str(row["Commit Info"]))
        except Exception:
            commits_list = []
        if len(commits_list) <= 1:
            continue
        updated_devs = {commit.get("commit_author", "Unknown") for commit in commits_list[1:]}
        for dev in updated_devs:
            developer_updated_prs[dev] = developer_updated_prs.get(dev, 0) + 1
    df_updated_prs = pd.DataFrame(list(developer_updated_prs.items()), columns=["Developer", "Developer_updated_MRs"])

    # ARs created by the creator (from Jira issues)
    creator_counts = df_issues_clean.groupby("Creator_User").size().reset_index(name="Creator_ARs")
    creator_counts.rename(columns={"Creator_User": "Developer"}, inplace=True)

    # ARs developed by developer (from issues assigned)
    developer_counts = df_issues_clean.groupby("Assignee_User").size().reset_index(name="Developer_ARs")
    developer_counts.rename(columns={"Assignee_User": "Developer"}, inplace=True)

    # ARs tested by tester (from issues reported)
    tester_counts = df_issues_clean.groupby("Reporter_User").size().reset_index(name="Tester_ARs")
    tester_counts.rename(columns={"Reporter_User": "Developer"}, inplace=True)

    # Developer fixed defects (number of fixed bugs by developer, assuming the assignee fixed them)
    df_fixed_bugs = df_issues_clean[
        (df_issues_clean["Issue Type"] == "Bug") &
        (df_issues_clean["Status"].isin(status_completion_keys))
        ].copy()
    df_fixed_bugs["Developer"] = df_fixed_bugs["Assignee_User"]
    df_fixed_counts = df_fixed_bugs.groupby("Developer").size().reset_index(name="Developer_fixed_defects")

    # Tester detected defects (bugs reported by testers)
    df_bugs = df_issues_clean[df_issues_clean["Issue Type"].isin(["Bug", "Defect"])].copy()
    df_bugs = df_bugs[df_bugs["Status"].isin(status_completion_keys)]
    df_reporter_counts = df_bugs.groupby("Reporter_User").size().reset_index(name="Tester_detected_defects")
    df_reporter_counts.rename(columns={"Reporter_User": "Developer"}, inplace=True)

    # Merge all features on Developer
    combined = dev_commits.merge(dev_created_mrs, on="Developer", how="outer")
    combined = combined.merge(dev_modified_files, on="Developer", how="outer")
    combined = combined.merge(dev_review_df, on="Developer", how="outer")
    combined = combined.merge(df_updated_prs, on="Developer", how="outer")
    combined = combined.merge(df_fixed_counts, on="Developer", how="outer")
    combined = combined.merge(df_reporter_counts, on="Developer", how="outer")
    combined = combined.merge(creator_counts, on="Developer", how="outer")
    combined = combined.merge(developer_counts, on="Developer", how="outer")
    combined = combined.merge(tester_counts, on="Developer", how="outer")
    combined.fillna(0, inplace=True)

    # Merge with the GitHub contributor list (which may include additional fields, e.g., overall contributions)
    df_contrib_renamed = df_contrib.copy().rename(columns={"Name": "Developer"})
    df_developer = df_contrib_renamed.merge(combined, on="Developer", how="outer")
    df_developer.fillna(0, inplace=True)

    df_developer.rename(columns={"Contributions": "Developer_Rank"}, inplace=True)

    return df_developer


# ==========================================================
# 6. Merge Developer Features with Issues
# ==========================================================
def merge_developer_features_with_issues(df_issues_clean, df_developer_features):
    # We want to join developer-level features with issues based on:
    # - Assignee_User -> Developer features for developers
    # - Reporter_User -> Tester features
    # - Creator_User -> Creator features
    df_features = df_developer_features.copy()

    # Merge for assignee features
    assignee_cols = [col for col in df_features.columns if col.startswith("Developer_")]
    assignee_merge = df_features[["Developer"] + assignee_cols].copy()
    assignee_merge.rename(columns={"Developer": "Assignee_User"}, inplace=True)
    df_merged = pd.merge(df_issues_clean, assignee_merge, how="left", on="Assignee_User")

    # Merge for reporter (tester) features: look for columns starting with "Tester_"
    tester_cols = [col for col in df_features.columns if col.startswith("Tester_")]
    tester_merge = df_features[["Developer"] + tester_cols].copy()
    tester_merge.rename(columns={"Developer": "Reporter_User"}, inplace=True)
    df_merged = pd.merge(df_merged, tester_merge, how="left", on="Reporter_User")

    # Merge for creator features: columns starting with "Creator_"
    creator_cols = [col for col in df_features.columns if col.startswith("Creator_")]
    creator_merge = df_features[["Developer"] + creator_cols].copy()
    creator_merge.rename(columns={"Developer": "Creator_User"}, inplace=True)
    df_merged = pd.merge(df_merged, creator_merge, how="left", on="Creator_User")

    df_merged.fillna(0, inplace=True)
    return df_merged


def encode_single_version(version_str):
    """
    Encode a single version string (e.g., "3.4.2") into a numeric value.
    This example uses a simple weighted sum of major, minor, and patch numbers.
    """
    try:
        parts = version_str.strip().split('.')
        # Convert parts to integers; if parts are missing, assume 0.
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        # Combine with weights: major has the highest weight.
        encoded = major + minor / 100 + patch / 10000
        return encoded
    except Exception as e:
        return 0.0  # Return 0.0 in case of any conversion error


def encode_versions(version_field):
    """
    Handle a version field that may contain multiple versions separated by commas.
    Converts the input to a string if necessary, then splits, encodes each version,
    and aggregates them (using the average).
    """
    # If the field is NaN, return 0.0
    if pd.isnull(version_field):
        return 0.0
    # Ensure we are working with a string
    version_field = str(version_field)
    version_list = version_field.split(',')
    encoded_versions = [encode_single_version(v) for v in version_list if v.strip() != '']
    if encoded_versions:
        aggregated_value = sum(encoded_versions) / len(encoded_versions)
    else:
        aggregated_value = 0.0
    return aggregated_value


def normalize_scale(df):
    # Encode Version from string to include it in scaling
    df["Version_encoded"] = df["Version"].apply(encode_versions)

    # Define the columns we want to use for each of the three aggregated features.
    features_creator = ["Creator_ARs", "Contributors"]
    features_assignee = ["Developer_ARs", "Developer_commits", "Developer_commits_reviews",
                         "Developer_modified_files", "Developer_created_MRs",
                         "Developer_updated_MRs", "Developer_fixed_defects", "Developer_Rank"]
    features_reporter = ["Tester_ARs", "Tester_detected_defects"]
    generic_features = ["Version_encoded"]

    # Normalize the selected columns to the range [0, 1] to ensure comparability.
    scaler = MinMaxScaler()
    df[features_creator + features_assignee + features_reporter + generic_features] = scaler.fit_transform(
        df[features_creator + features_assignee + features_reporter + generic_features]
    )

    # scaler = MinMaxScaler()
    # df[features_creator + features_assignee + features_reporter] = scaler.fit_transform(
    #    df[features_creator + features_assignee + features_reporter]
    # )

    # Aggregate features:
    # Here, we take the average of the normalized values for each group.
    df["Creator_count"] = df[features_creator].mean(axis=1)
    df["Assignee_count"] = df[features_assignee].mean(axis=1)
    df["Reporter_count"] = df[features_reporter].mean(axis=1)

    # Naming feature same as FINE-SE
    df = df[["Summary", "Story Points", "Assignee_count", "Reporter_count", "Creator_count"]].copy()
    df.rename(columns={"Story Points": "Custom field (Story Points)"}, inplace=True)
    # Save the aggregated expert features for use in further modeling.
    return df  # df.to_csv("aggregated_expert_features.csv", index=False)


# def remove_outliers(df):
#     # Identify numeric columns only.
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#
#     # Compute Q1, Q3, and IQR for those columns.
#     Q1 = df[numeric_cols].quantile(0.25)
#     Q3 = df[numeric_cols].quantile(0.75)
#     IQR = Q3 - Q1
#
#     # Build a mask of rows that are *not* outliers in *any* numeric column.
#     mask = ~(
#             (df[numeric_cols] < (Q1 - 1.5 * IQR)) |
#             (df[numeric_cols] > (Q3 + 1.5 * IQR))
#     ).any(axis=1)
#
#     # Apply that mask to the entire df, not just the numeric subset.
#     df_no_outliers = df[mask].copy()
#
#     # Now df_no_outliers retains all columns (numeric + non-numeric),
#     # but only the rows that passed your outlier criterion.
#     print("Original shape:", df.shape)
#     print("Shape after outlier removal:", df_no_outliers.shape)
#     return df_no_outliers

def remove_outliers(df):
    # Identify numeric columns only.
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Compute Q1, Q3, and IQR for those columns.
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers for each column.
    outlier_counts = {}
    for col in numeric_cols:
        is_outlier = (df[col] < (Q1[col] - 1.5 * IQR[col])) | (df[col] > (Q3[col] + 1.5 * IQR[col]))
        outlier_counts[col] = is_outlier.sum()

    # Build a mask of rows that are *not* outliers in *any* numeric column.
    mask = ~(
            (df[numeric_cols] < (Q1 - 1.5 * IQR)) |
            (df[numeric_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)

    # Apply that mask to the entire df, not just the numeric subset.
    df_no_outliers = df[mask].copy()

    # Calculate overall number of removed outliers.
    total_outliers = len(df) - len(df_no_outliers)

    # Print summary
    print(f"\nTotal outliers removed: {total_outliers}")
    print(f"Original shape: {df.shape}")
    print(f"Shape after outlier removal: {df_no_outliers.shape}")

    return df_no_outliers
