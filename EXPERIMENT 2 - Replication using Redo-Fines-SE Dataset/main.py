import pandas as pd

import fetch as fetch
import process as process
import json
import os


def process_project(project_name, config, token, issue_completion_key):
    # Create an output directory for the project.
    jira_domain = config["jira_domain"]
    jira_project_key = config["jira_project_key"]
    github_repo = config["github_repos"]
    output_dir = config["output_dir"]
    story_point_field = config["story_point_field"]
    jira_url_version = config["jira_url_version"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n=== Processing Project: {project_name} ===")
    print(f"GitHub Repo: {github_repo}, Jira Project Key: {jira_project_key}")

    # Step 1: Fetch GitHub contributors
    if os.path.exists(os.path.join(output_dir, "contributors.xlsx")):
        df_contrib = pd.read_excel(os.path.join(output_dir, "contributors.xlsx"))
    else:
        df_contrib = fetch.github_contributors(github_repo, output_dir, token)

    # Step 2: Fetch Jira issues
    if os.path.exists(os.path.join(output_dir, "issues.xlsx")):
        df_issues = pd.read_excel(os.path.join(output_dir, "issues.xlsx"))
    else:
        df_issues = fetch.jira_issues(jira_domain, jira_url_version, jira_project_key, output_dir, story_point_field)

    # Step 3: Fetch GitHub PR data
    if os.path.exists(os.path.join(output_dir, "pull_requests.xlsx")):
        df_pr = pd.read_excel(os.path.join(output_dir, "pull_requests.xlsx"))
    else:
        df_pr = fetch.github_pr(github_repo, output_dir, token)

    # Step 4: Clean Jira issues data and add the 'Contributors' feature
    df_issues_clean = process.clean_issue_data(df_issues, output_dir, jira_project_key, issue_completion_key)

    # Step 5: Aggregate developer features
    df_developer_features = process.aggregate_developer_features(df_pr, df_issues_clean, df_contrib,
                                                                 issue_completion_key)

    dev_features_path = os.path.join(output_dir, "developer_activity_features.xlsx")
    df_developer_features.to_excel(dev_features_path, index=False)
    print(f"Saved developer features to: {dev_features_path}")

    # Step 6: Merge developer features with issues data
    df_merged = process.merge_developer_features_with_issues(df_issues_clean, df_developer_features)
    merged_path = os.path.join(output_dir, "issues_developer_features_merged.xlsx")

    # Drop columns that are not needed:
    df_merged.drop(columns=["Issue Key", "Issue Type", "Description", "Status", "Created", "Resolved"], inplace=True)
    df_merged.to_excel(merged_path, index=False)
    print(f"Full dataset for {project_name} containing 13 dev features and summary saved to: {merged_path}")

    # Remove outliers
    df_no_outliers = process.remove_outliers(df_merged)

    df_final = process.normalize_scale(df_no_outliers)
    final_path = os.path.join(output_dir, f"{project_name}.csv")
    print(f"Final dataset for {project_name} saved to: {final_path}")
    df_final.to_csv(final_path, index=False)


'''
def process_project(project_name, config, github_token=None):
    """
    Orchestrates fetching JIRA issues, GitHub data, cleaning, building user features,
    merging them, etc., for a single project.
    """
    jira_domain = config["jira_domain"]
    jira_project_key = config["jira_project_key"]
    github_repo = config["github_repo"]
    output_dir = config["output_dir"]

    # 1) Fetch JIRA
    df_issues_raw = fetch.fetch_jira_issues(
        jira_domain=jira_domain,
        project_key=jira_project_key,
        output_dir=output_dir,
        token=None  # or your JIRA token
    )

    # 2) Clean
    df_issues_clean = process.clean_jira_issues(df_issues_raw, output_dir)

    # 3) Fetch GitHub
    df_contrib = fetch.fetch_github_contributors_data(
        github_repo=github_repo,
        output_dir=output_dir,
        token=github_token
    )

    # 4) Build user features
    df_user_features = process.build_user_features(df_issues_clean, df_contrib, output_dir)

    # 5) Merge role-based features onto issues
    df_merged = process.merge_role_features(df_issues_clean, df_user_features, output_dir)

    print(f"Done processing {project_name}. Output in {output_dir}.")
'''

if __name__ == "__main__":

    with open("projects_config.json", "r") as f:
        PROJECT_CONFIGS = json.load(f)

    GITHUB_TOKEN = PROJECT_CONFIGS['TOKEN']  # or read from env variable
    ISSUE_COMPLETION_KEYS = PROJECT_CONFIGS['ISSUE_COMPLETION_KEY']

    for project_name, config in {k: v for k, v in PROJECT_CONFIGS.items() if k not in ["TOKEN",
                                                                                       "ISSUE_COMPLETION_KEY",
                                                                                       "TALEND_DQ"]}.items():
        process_project(project_name, config, token=GITHUB_TOKEN, issue_completion_key=ISSUE_COMPLETION_KEYS)
