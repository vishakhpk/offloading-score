"""Read issue.json from folder, fetch issue body from GitHub, and summarize with LLM."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import requests

# Initialize package (sets up sys.path)
_init_path = Path(__file__).parent / "__init__.py"
exec(_init_path.read_text())

from utils import call_openai, download_and_encode_image, load_json, save_json
from dotenv import load_dotenv

load_dotenv()


def discover_issue_dirs(parent_dir: Path) -> list[Path]:
    """Discover all issue directories in the parent directory."""
    parent_dir = parent_dir.expanduser().resolve()
    if not parent_dir.is_dir():
        raise SystemExit(f"Parent directory not found: {parent_dir}")
    
    issue_dirs = []
    for child in sorted(parent_dir.iterdir()):
        if not child.is_dir() or not child.name.lower().startswith("issue"):
            continue
        issue_json_path = child / "issue.json"
        if issue_json_path.exists():
            issue_dirs.append(child)
        else:
            print(f"Skipping {child.name}: no issue.json found")
    
    return issue_dirs


def load_issue_json(issue_dir: Path) -> dict | None:
    """Load issue.json from the issue directory."""
    issue_json_path = issue_dir / "issue.json"
    if not issue_json_path.exists():
        return None
    
    try:
        return load_json(issue_json_path)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading issue.json from {issue_dir}: {e}")
        return None


def normalize_issue_api_url(issue_url: str) -> str:
    """Convert GitHub issue URL to API URL."""
    issue_url = issue_url.strip()
    if not issue_url:
        raise ValueError("Issue URL is empty")
    
    if "api.github.com" in issue_url:
        return issue_url
    
    match = re.search(r"github\.com/([^/]+)/([^/]+)/issues/(\d+)", issue_url)
    if not match:
        raise ValueError(f"Unrecognized GitHub issue URL: {issue_url}")
    
    owner, repo, number = match.groups()
    return f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"


def fetch_issue_from_github(issue_api_url: str) -> dict:
    """Fetch issue data from GitHub API."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "specstory-preprocessing",
    }
    token = os.getenv("GIT_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    response = requests.get(issue_api_url, headers=headers, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"GitHub request failed ({response.status_code}): {response.text[:200]}")
    
    return response.json()


def extract_image_urls(text: str | None) -> list[str]:
    """Extract image URLs from markdown text."""
    if not text:
        return []
    
    # Match markdown image syntax: ![alt](url) or just (url) for images
    image_pattern = r'!\[.*?\]\((https?://[^\s\)]+)\)|\((https?://[^\s\)]+\.(?:png|jpg|jpeg|gif|webp|svg))\)'
    matches = re.findall(image_pattern, text, re.IGNORECASE)
    
    # Extract URLs from matches (handles both groups)
    urls = []
    for match in matches:
        url = match[0] if match[0] else match[1]
        if url:
            urls.append(url)
    
    # Also look for raw image URLs
    raw_url_pattern = r'https?://[^\s\)]+\.(?:png|jpg|jpeg|gif|webp|svg)'
    raw_urls = re.findall(raw_url_pattern, text, re.IGNORECASE)
    urls.extend(raw_urls)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls


def summarize_with_llm(body: str | None, image_urls: list[str] | None = None, model_name: str = "gpt-4o") -> str:
    """Summarize issue body using LLM, including images if provided."""
    if not body or not body.strip():
        return ""
    
    prompt = (
        "Summarize the GitHub issue."
        "Describe the task that the contributor is asked to complete. Be specific in the description."
    )
    
    # Build content as a list (can include text and images)
    content: list[dict[str, Any]] = [{"type": "text", "text": body.strip()}]
    
    # Download and encode images if available
    if image_urls:
        content.append({"type": "text", "text": "\n\nImages referenced in this issue:"})
        for i, url in enumerate(image_urls, 1):
            encoded = download_and_encode_image(url)
            if encoded:
                content.append({"type": "text", "text": f"Image {i}:"})
                content.append(encoded)
            else:
                # Fallback to URL if download fails
                content.append({"type": "text", "text": f"Image {i} (URL): {url}"})
    
    summary = call_openai(prompt=prompt, content=content, model_name=model_name)
    return summary.strip() if summary else ""


def update_issue_json(issue_dir: Path, summary: dict) -> None:
    """Update issue.json with summary data."""
    issue_json_path = issue_dir / "issue.json"
    existing = load_issue_json(issue_dir)
    
    if existing is None:
        existing = {}
    
    # Merge summary data into existing data
    existing.update(summary)
    
    save_json(existing, issue_json_path)
    print(f"Updated {issue_json_path}")


def process_issue(issue_dir: Path, model_name: str, output_field: str) -> bool:
    """Process a single issue: fetch from GitHub and summarize."""
    try:
        # Load issue.json
        issue_data = load_issue_json(issue_dir)
        if issue_data is None:
            print(f"Skipping {issue_dir.name}: could not load issue.json")
            return False
        
        issue_url = issue_data.get("issue_url")
        if not issue_url:
            print(f"Skipping {issue_dir.name}: no issue_url in issue.json")
            return False
        
        # Fetch issue from GitHub
        issue_api_url = normalize_issue_api_url(issue_url)
        github_data = fetch_issue_from_github(issue_api_url)
        
        # Extract image URLs from issue body
        body = github_data.get("body")
        image_urls = extract_image_urls(body)
        
        if image_urls:
            print(f"Found {len(image_urls)} image(s) in issue: {', '.join(image_urls[:3])}{'...' if len(image_urls) > 3 else ''}")
        
        # Summarize body with LLM (including image URLs)
        summary_text = summarize_with_llm(body, image_urls, model_name)
        
        # Update issue.json with summary
        summary_data = {
            output_field: summary_text,
            "title": github_data.get("title"),
            "labels": [label.get("name") for label in github_data.get("labels", []) if isinstance(label, dict)],
        }
        
        # Include image URLs if found
        if image_urls:
            summary_data["image_urls"] = image_urls
        
        update_issue_json(issue_dir, summary_data)
        if summary_text:
            preview = summary_text[:200] + "..." if len(summary_text) > 200 else summary_text
            print(f"Summary for {issue_dir.name}: {preview}\n")
        return True
    except Exception as e:
        print(f"Error processing {issue_dir.name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read issue.json from folders, fetch issue body from GitHub, and summarize with LLM."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to parent directory containing issue_* folders",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use for summarization (default: gpt-4o)",
    )
    parser.add_argument(
        "--output-field",
        default="body_summary",
        help="Field name to store summary in issue.json (default: body_summary)",
    )
    args = parser.parse_args()
    
    parent_dir = args.data.expanduser().resolve()
    if not parent_dir.is_dir():
        raise SystemExit(f"Parent directory not found: {parent_dir}")
    
    # Discover all issue directories with issue.json files
    issue_dirs = discover_issue_dirs(parent_dir)
    
    if not issue_dirs:
        raise SystemExit(f"No issue directories with issue.json found in {parent_dir}")
    
    print(f"Found {len(issue_dirs)} issue folder(s) to process\n")
    
    # Process each issue
    success_count = 0
    for issue_dir in issue_dirs:
        print(f"Processing {issue_dir.name}...")
        if process_issue(issue_dir, args.model, args.output_field):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(issue_dirs)} issues processed successfully")


if __name__ == "__main__":
    main()
