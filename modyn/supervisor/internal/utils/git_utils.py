import subprocess


def get_head_sha() -> str:
    """When launched in a git repository, return the SHA of the current HEAD."""

    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    assert result.returncode == 0, f"Failed to get HEAD SHA: {result.stderr}, {result.stdout}"
    return result.stdout.strip()
