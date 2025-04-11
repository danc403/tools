import requests
import json

# Replace with your GitHub PAT
GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"

# Replace with the username of the person you want to add
USERNAME_TO_ADD = "THE_USERNAME_TO_ADD"

# List of repositories (replace with your repository names)
REPOSITORIES = [
    "your-username/repo1",
    "your-username/repo2",
    "another-username/some-repo" # You can add to repos you admin
]

# Desired permission level: "admin", "push", or "pull"
PERMISSION = "admin"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

for repo in REPOSITORIES:
    owner, repo_name = repo.split("/")
    url = f"https://api.github.com/repos/{owner}/{repo_name}/collaborators/{USERNAME_TO_ADD}"
    data = {"permission": PERMISSION}

    response = requests.put(url, headers=headers, data=json.dumps(data))

    if response.status_code == 204:
        print(f"Successfully added {USERNAME_TO_ADD} as a {PERMISSION} collaborator to {repo}")
    elif response.status_code == 201:
        print(f"Successfully invited {USERNAME_TO_ADD} as a {PERMISSION} collaborator to {repo}")
    else:
        print(f"Failed to add {USERNAME_TO_ADD} to {repo}. Status code: {response.status_code}, Response: {response.text}")

print("Script finished.")
