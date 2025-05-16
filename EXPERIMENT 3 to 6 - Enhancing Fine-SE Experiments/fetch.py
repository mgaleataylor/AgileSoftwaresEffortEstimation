import requests
import pandas as pd
import os

'''
def github_contributors(github_repo, output_dir, token):
    """
    Fetches contributors, PRs, commits, etc. for a given GitHub repo.
    Returns multiple DataFrames or writes them to files.
    """

    base_url = f"https://api.github.com/repos/{github_repo}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}"
    }

    # 1) Fetch contributors
    contributors_list = []
    page = 1
    while True:
        contrib_url = f"{base_url}/contributors?per_page=100&page={page}"
        resp = requests.get(contrib_url, headers=headers)
        data = resp.json()
        if not data or "message" in data:
            break
        contributors_list.extend(data)
        page += 1
    df_contrib = pd.DataFrame([{
        "ID": c.get("id"),
        "Name": c.get("name"),
        "Contributions": c.get("contributions"),
        "Type": c.get("type")
    } for c in contributors_list])

    df_contrib.to_excel(os.path.join(output_dir, "contributors.xlsx"), index=False)

    return df_contrib
'''


def fetch_org_repos(org, token, per_page=100):
    """
    Fetch all repositories for a given GitHub organization.

    Args:
        org (str): The GitHub organization name.
        token (str): Your GitHub personal access token.
        per_page (int): Number of repos per page (max 100).

    Returns:
        List[str]: A list of repository identifiers (e.g., "org/repoName").
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}"
    }
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/orgs/{org}/repos?per_page={per_page}&page={page}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching page {page}: {response.status_code} {response.text}")
            break
        data = response.json()
        if not data:
            break
        for repo in data:
            repos.append(repo.get("full_name"))  # full_name is "org/repo"
        page += 1
    return repos


def drop_duplicate_contributors(df_contributors):
    """
    Drop duplicates but also sums the contributions
    """
    df = pd.DataFrame(df_contributors, columns=["Repo", "ID", "Login", "Name", "Contributions", "Type"])

    # Optional: strip whitespace from Login if needed
    df["Login"] = df["Login"].astype(str).str.strip()
    # Group by the 'Login' column and sum the Contributions
    df_grouped = df.groupby("Login", as_index=False)["Contributions"].sum()
    # Get one row per contributor for extra details (using the first occurrence)
    df_details = df.drop_duplicates(subset=["Login"])[["Login", "Name", "Type"]]

    # Merge the summed contributions with the contributor details
    df_final = pd.merge(df_details, df_grouped, on="Login", how="left")
    return df_final


def github_contributors(github_repos, output_dir, token):
    """
    Fetches contributors for each GitHub repository in `github_repos`.
    For each contributor, fetches additional user details (like full name) from the GitHub API.
    Returns a combined DataFrame of contributor information and writes it to an Excel file.

    Parameters:
        github_repos (list or str): List of repository identifiers, e.g., ["apache/mesos", "org/repo2", ...]
                                   OR a single organization name.
        output_dir (str): Directory to save the output file.
        token (str): GitHub personal access token.
    """
    import time  # for sleep between requests
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}"
    }

    all_contributors_details = []
    all_repos = [github_repos]
    # If github_repos doesn't look like a repo string (i.e. it lacks a '/'),
    # assume it's an organization name and fetch its repos.
    if '/' not in github_repos:
        all_repos = fetch_org_repos(org=github_repos, token=token)

    for repo in all_repos:
        base_url = f"https://api.github.com/repos/{repo}"
        print(f"Processing repository: {repo}")

        # 1) Fetch contributors from the repo
        contributors_list = []
        page = 1
        while True:
            contrib_url = f"{base_url}/contributors?per_page=100&page={page}"
            resp = requests.get(contrib_url, headers=headers)
            if resp.status_code != 200:
                print(f"Error fetching contributors from {repo} page {page}: {resp.status_code} {resp.text}")
                break

            # Check if response content is empty.
            if not resp.text.strip():
                print(f"No content in response from {repo} page {page}.")
                break

            try:
                data = resp.json()
            except Exception as e:
                print(f"Error decoding JSON for {repo} page {page}: {e}")
                break

            # If data is empty or has an error message, break out.
            if not data or ("message" in data and data["message"]):
                break

            contributors_list.extend(data)
            page += 1
            time.sleep(0.5)  # optional: to avoid hitting rate limits

        # 2) For each contributor, fetch additional user details using their login.
        for c in contributors_list:
            login = c.get("login")
            user_url = f"https://api.github.com/users/{login}"
            user_resp = requests.get(user_url, headers=headers)
            if user_resp.status_code != 200:
                print(f"Error fetching user details for {login}: {user_resp.status_code} {user_resp.text}")
                full_name = login
            else:
                try:
                    user_data = user_resp.json()
                    full_name = user_data.get("name") or login
                except Exception as e:
                    print(f"Error decoding JSON for user {login}: {e}")
                    full_name = login

            contributor_detail = {
                "Repo": repo,
                "ID": c.get("id"),
                "Login": login,
                "Name": full_name,
                "Contributions": c.get("contributions"),
                "Type": c.get("type")
            }
            all_contributors_details.append(contributor_detail)
            time.sleep(0.2)

    df_contrib = pd.DataFrame(all_contributors_details)
    df_contrib = drop_duplicate_contributors(df_contrib)
    save_path = os.path.join(output_dir, "contributors.xlsx")
    df_contrib.to_excel(save_path, index=False)
    print(f"Saved GitHub contributors to: {save_path}")
    return df_contrib


def github_pr(github_repos, output_dir, token):
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}"
    }
    all_pull_data = []

    all_repos = [github_repos]
    # If github_repos doesn't look like a repo string (contains a slash),
    # assume it's an org name and fetch its repos.
    if '/' not in github_repos:
        all_repos = fetch_org_repos(org=github_repos, token=token)

    # Loop over each repository.
    for repo in all_repos:
        base_url = f"https://api.github.com/repos/{repo}"
        print(f"Processing PRs for repository: {repo}")
        page = 1
        per_page = 50

        while True:
            pr_url = f"{base_url}/pulls?state=all&per_page={per_page}&page={page}"
            pr_response = requests.get(pr_url, headers=headers)
            if pr_response.status_code != 200:
                print(f"Error fetching PRs from {repo} page {page}: {pr_response.status_code} {pr_response.text}")
                break

            pull_requests = pr_response.json()
            if not pull_requests or "message" in pull_requests:
                # No more pull requests for this repo
                break

            for pr in pull_requests:
                pr_number = pr.get("number")
                pr_login = pr.get("user", {}).get("login", "Unknown")

                # Fetch full user details for the PR creator.
                user_url = f"https://api.github.com/users/{pr_login}"
                user_resp = requests.get(user_url, headers=headers)
                if user_resp.status_code == 200:
                    user_data = user_resp.json()
                    pr_user_full = user_data.get("name") or pr_login
                else:
                    pr_user_full = pr_login

                # Fetch commits for the PR.
                commits_url = f"{base_url}/pulls/{pr_number}/commits"
                commits_resp = requests.get(commits_url, headers=headers)
                commits = commits_resp.json() if commits_resp.status_code == 200 else []
                commit_info = [{
                    "commit_hash": commit.get("sha"),
                    "commit_author": commit.get("commit", {}).get("author", {}).get("name", "Unknown")
                } for commit in commits]
                commit_count = len(commit_info)

                # Fetch reviews for the PR.
                reviews_url = f"{base_url}/pulls/{pr_number}/reviews"
                reviews_resp = requests.get(reviews_url, headers=headers)
                reviews = reviews_resp.json() if reviews_resp.status_code == 200 else []
                reviews_info = [{
                    "Review ID": review.get("id"),
                    "Reviewer": review.get("user", {}).get("login"),
                    "Review State": review.get("state"),
                    "Review Submitted At": review.get("submitted_at")
                } for review in reviews]

                # Fetch files modified in the PR.
                files_url = f"{base_url}/pulls/{pr_number}/files"
                files_resp = requests.get(files_url, headers=headers)
                files = files_resp.json() if files_resp.status_code == 200 else []
                files_modified = len(files)

                pull_data = {
                    "Repo": repo,
                    "PR Number": pr_number,
                    "PR User": pr_user_full,
                    "Commits": commit_count,
                    "Commit Info": commit_info,
                    "Reviewers Info": reviews_info,
                    "Files Modified": files_modified
                }
                all_pull_data.append(pull_data)

            print(f"Processed PR page {page} for repo {repo} (total PRs so far: {len(all_pull_data)})")
            page += 1

    # Create a DataFrame once all repos have been processed.
    df = pd.DataFrame(all_pull_data)
    save_path = os.path.join(output_dir, "pull_requests.xlsx")
    df.to_excel(save_path, index=False)
    print(f"Saved PR data to: {save_path}")
    return df


def jira_issues(jira_domain, jira_url_version, project_key, output_dir, story_point_field, token=None, max_results=100):
    """
    Fetches all issues from a given JIRA project, saves them to an Excel or CSV file,
    and returns a DataFrame.
    """

    all_issues = []
    start_at = 0
    jql = f"project={project_key}"

    # If you need Basic Auth or a JIRA token, adapt accordingly:
    headers = {
        "Accept": "application/json"
    }
    # If token is needed, you might do headers["Authorization"] = f"Bearer {token}"

    while True:
        # Adjust for your JIRA version (2 vs 3). For example:
        endpoint = f"https://{jira_domain}/rest/api/{jira_url_version}/search"
        params = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": max_results
        }

        response = requests.get(endpoint, headers=headers, params=params)
        data = response.json()

        issues = data.get("issues", [])
        if not issues:
            break

        for issue in issues:
            fields = issue.get("fields", {})
            fix_versions = fields.get("versions", [])
            fix_version_names = [fv.get("name") for fv in fix_versions if fv.get("name")]
            fix_version_str = ",".join(fix_version_names)
            # ... gather relevant fields ...
            all_issues.append({
                "Issue Key": issue.get("key"),
                "Issue Type": fields.get("issuetype", {}).get("name", ""),
                "Summary": fields.get("summary", ""),
                "Description": fields.get("description", ""),
                "Status": fields.get("status", {}).get("name", ""),
                "Created": fields.get("created", ""),
                "Resolved": fields.get("resolutiondate", ""),
                "Story Points": fields.get(story_point_field) or {},
                "Assignee_User": (fields.get("assignee") or {}).get("displayName", ""),
                "Reporter_User": (fields.get("reporter") or {}).get("displayName", ""),
                "Creator_User": (fields.get("creator") or {}).get("displayName", ""),
                "Version": fix_version_str
            })

        start_at += len(issues)
        if start_at >= data.get("total", 0):
            break

    df_issues = pd.DataFrame(all_issues)
    outfile = os.path.join(output_dir, "issues.xlsx")
    df_issues.to_excel(outfile, index=False)

    return df_issues
