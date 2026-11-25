import base64
import hashlib
import json
import re
import time
import tomllib
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlparse

import httpx

from backend.app.config import settings
from backend.app.schemas import GitHubProfile, RepoEvidence, UnlinkedIdentity

GITHUB_API_BASE = "https://api.github.com"
MAX_REPOS = 15
MAX_TREE_PATHS = 300
MAX_README_CHARS = 3000
MAX_COMMITS_CAP = 300
MAX_SAMPLED_COMMITS = 30
MAX_RATE_WAIT_SECONDS = 60
MAX_JSON_STATS_BYTES = 100 * 1024

SKIP_DIRS = {"node_modules", "venv", ".venv", "dist", ".git"}
STATS_EXCLUDED_DIRS = {"vendor", "node_modules", "dist", "build", ".venv", "venv", ".git"}
LOCKFILES = {
    "package-lock.json",
    "yarn.lock",
    "composer.lock",
    "poetry.lock",
    "pnpm-lock.yaml",
    "cargo.lock",
}
GENERATED_SUFFIXES = (".min.js", ".min.css", ".min.mjs", ".bundle.js", ".map")
BINARY_OR_DATA_EXTS = {
    ".csv",
    ".pt",
    ".pth",
    ".onnx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp3",
    ".mp4",
    ".wav",
    ".flac",
    ".npy",
    ".npz",
    ".joblib",
    ".pkl",
    ".pickle",
    ".sqlite",
    ".db",
}
DEPENDENCY_FILES = {"requirements.txt", "package.json", "pyproject.toml", "composer.json"}


def _build_headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "knock-github-analyzer",
    }
    token = settings.github_token.strip()
    if token and not token.startswith("your_"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_with_rate_limit(
    client: httpx.Client,
    url: str,
    params: dict[str, str | int] | None = None,
    accept: str = "application/vnd.github+json",
) -> httpx.Response:
    headers = _build_headers(accept=accept)
    for attempt in range(2):
        response = client.get(url, params=params, headers=headers)
        if response.status_code in (403, 429):
            remaining = response.headers.get("X-RateLimit-Remaining")
            retry_after = response.headers.get("Retry-After")
            reset_ts = response.headers.get("X-RateLimit-Reset")

            is_rate_limited = response.status_code == 429 or remaining == "0" or retry_after is not None
            if not is_rate_limited:
                return response

            wait_seconds = 0
            if retry_after and retry_after.isdigit():
                wait_seconds = int(retry_after)
            elif reset_ts and reset_ts.isdigit():
                wait_seconds = max(0, int(reset_ts) - int(time.time()) + 1)

            if attempt == 0 and 0 < wait_seconds <= MAX_RATE_WAIT_SECONDS:
                time.sleep(wait_seconds)
                continue

            reset_info = f" (resets in {wait_seconds}s)" if wait_seconds > 0 else ""
            raise RuntimeError(
                f"GitHub API rate limit exceeded{reset_info}. Configure GITHUB_TOKEN in .env."
            )
        return response
    return response


def fetch_file(full_name: str, path: str, max_bytes: int = 20000) -> str | None:
    clean_path = path.lstrip("/")
    if not clean_path:
        return None
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        return _fetch_file_with_client(client, full_name, clean_path, max_bytes=max_bytes)


def _fetch_file_with_client(
    client: httpx.Client,
    full_name: str,
    clean_path: str,
    max_bytes: int = 20000,
) -> str | None:
    url = f"{GITHUB_API_BASE}/repos/{full_name}/contents/{clean_path}"
    response = _request_with_rate_limit(client, url)
    if response.status_code in (404, 409):
        return None
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or data.get("type") != "file":
        return None

    if data.get("encoding") == "base64" and data.get("content"):
        raw = base64.b64decode(data["content"])[:max_bytes]
        return raw.decode("utf-8", errors="replace")

    download_url = data.get("download_url")
    if download_url:
        dl = client.get(download_url, headers=_build_headers(accept="*/*"))
        if dl.status_code == 200:
            return dl.content[:max_bytes].decode("utf-8", errors="replace")
    return None


def _parse_github_repo_link(link: str) -> tuple[str, str] | None:
    raw = link.strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.netloc.lower() not in ("github.com", "www.github.com"):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None
    return owner, repo


def _parse_requirements_txt(content: str) -> list[str]:
    deps: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if match:
            deps.append(match.group(1))
    return deps


def _parse_package_json(content: str) -> list[str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    deps: list[str] = []
    for key in ("dependencies", "devDependencies"):
        section = data.get(key)
        if isinstance(section, dict):
            deps.extend(str(k) for k in section.keys())
    return deps


def _parse_pyproject_toml(content: str) -> list[str]:
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return []
    deps: list[str] = []
    project = data.get("project")
    if isinstance(project, dict):
        for item in project.get("dependencies", []) or []:
            if isinstance(item, str):
                match = re.match(r"^\s*([A-Za-z0-9_.\-]+)", item)
                if match:
                    deps.append(match.group(1))
        opt = project.get("optional-dependencies")
        if isinstance(opt, dict):
            for group in opt.values():
                if isinstance(group, list):
                    for item in group:
                        if isinstance(item, str):
                            match = re.match(r"^\s*([A-Za-z0-9_.\-]+)", item)
                            if match:
                                deps.append(match.group(1))
    poetry = (data.get("tool") or {}).get("poetry")
    if isinstance(poetry, dict):
        poetry_deps = poetry.get("dependencies")
        if isinstance(poetry_deps, dict):
            for name in poetry_deps.keys():
                if str(name).lower() != "python":
                    deps.append(str(name))
    return deps


def _parse_composer_json(content: str) -> list[str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    deps: list[str] = []
    for key in ("require", "require-dev"):
        section = data.get(key)
        if isinstance(section, dict):
            deps.extend(str(k) for k in section.keys())
    return deps


def _extract_dependencies(client: httpx.Client, full_name: str, file_tree: list[str]) -> list[str]:
    candidates = [
        p for p in file_tree
        if PurePosixPath(p).name in DEPENDENCY_FILES
    ]
    candidates.sort(key=lambda p: (p.count("/"), p))
    candidates = candidates[:6]

    seen: set[str] = set()
    all_deps: list[str] = []
    for path in candidates:
        content = _fetch_file_with_client(client, full_name, path, max_bytes=50000)
        if not content:
            continue
        name = PurePosixPath(path).name
        if name == "requirements.txt":
            parsed = _parse_requirements_txt(content)
        elif name == "package.json":
            parsed = _parse_package_json(content)
        elif name == "pyproject.toml":
            parsed = _parse_pyproject_toml(content)
        elif name == "composer.json":
            parsed = _parse_composer_json(content)
        else:
            parsed = []

        for dep in parsed:
            if dep not in seen:
                seen.add(dep)
                all_deps.append(dep)
    return all_deps


def _sample_evenly(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return items
    step = (len(items) - 1) / (limit - 1)
    return [items[round(i * step)] for i in range(limit)]


def _hash_email(email: str) -> str:
    digest = hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


def _is_excluded_from_line_stats(file_entry: dict, file_sizes: dict[str, int]) -> bool:
    filename = file_entry.get("filename") or ""
    if not filename:
        return True

    posix_path = PurePosixPath(filename)
    basename = posix_path.name.lower()
    if basename in LOCKFILES:
        return True

    dir_parts = {p.lower() for p in posix_path.parts[:-1]}
    if dir_parts & STATS_EXCLUDED_DIRS:
        return True

    if any(basename.endswith(suffix) for suffix in GENERATED_SUFFIXES):
        return True
    if ".generated." in basename or "_generated." in basename or basename.endswith(".pb.go"):
        return True

    ext = posix_path.suffix.lower()
    if ext in BINARY_OR_DATA_EXTS:
        return True

    patch = file_entry.get("patch")
    if ext == ".json":
        known_size = file_sizes.get(filename)
        if known_size is not None and known_size > MAX_JSON_STATS_BYTES:
            return True
        if patch is None:
            return True
        if len(patch.encode("utf-8")) > MAX_JSON_STATS_BYTES:
            return True

    if patch is None and int(file_entry.get("changes") or 0) == 0:
        return True

    return False


def _analyze_commits(
    client: httpx.Client,
    full_name: str,
    username: str,
    confirmed_emails: set[str],
    file_sizes: dict[str, int],
) -> dict:
    all_commits: list[dict] = []
    for page in range(1, (MAX_COMMITS_CAP // 100) + 1):
        url = f"{GITHUB_API_BASE}/repos/{full_name}/commits"
        response = _request_with_rate_limit(
            client,
            url,
            params={"per_page": 100, "page": page},
        )
        if response.status_code in (404, 409):
            break
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            break
        all_commits.extend(batch)
        if len(batch) < 100 or len(all_commits) >= MAX_COMMITS_CAP:
            break

    all_commits = all_commits[:MAX_COMMITS_CAP]
    total_commit_count = len(all_commits)
    target = username.lower()

    known_user_emails: set[str] = set(confirmed_emails)
    for commit in all_commits:
        gh_login = ((commit.get("author") or {}).get("login") or "").lower()
        if gh_login == target:
            raw_email = (((commit.get("commit") or {}).get("author") or {}).get("email") or "").strip().lower()
            if raw_email:
                known_user_emails.add(raw_email)

    user_commits: list[dict] = []
    user_dates: list[str] = []
    user_commit_count = 0
    unlinked_commit_count = 0
    other_author_count = 0
    confirmed_email_commit_count = 0
    unlinked_raw: list[tuple[str, str]] = []

    for commit in all_commits:
        gh_login = ((commit.get("author") or {}).get("login") or "").strip()
        commit_meta = commit.get("commit") or {}
        git_author = commit_meta.get("author") or {}
        git_name = (git_author.get("name") or "").strip()
        git_email = (git_author.get("email") or "").strip()
        git_email_lower = git_email.lower()
        ts = git_author.get("date") or (commit_meta.get("committer") or {}).get("date")

        is_login_match = gh_login.lower() == target
        is_email_confirmed = bool(git_email_lower and git_email_lower in confirmed_emails)

        if not gh_login:
            unlinked_commit_count += 1
            unlinked_raw.append((git_name, git_email))

        if is_email_confirmed:
            confirmed_email_commit_count += 1

        if is_login_match or (not gh_login and is_email_confirmed):
            user_commit_count += 1
            user_commits.append(commit)
            if isinstance(ts, str) and ts:
                user_dates.append(ts)
        elif gh_login and not is_login_match:
            if is_email_confirmed:
                user_commit_count += 1
                user_commits.append(commit)
                if isinstance(ts, str) and ts:
                    user_dates.append(ts)
            else:
                other_author_count += 1

    seen_identities: set[tuple[str, str | None]] = set()
    unlinked_identities: list[UnlinkedIdentity] = []
    for name, email in unlinked_raw:
        email_lower = email.lower()
        if not email:
            stored_email = None
        elif email_lower in known_user_emails:
            stored_email = email
        else:
            stored_email = _hash_email(email)

        key = (name, stored_email)
        if key not in seen_identities:
            seen_identities.add(key)
            unlinked_identities.append(UnlinkedIdentity(name=name, email=stored_email))

    first_commit_at = min(user_dates) if user_dates else None
    last_commit_at = max(user_dates) if user_dates else None
    active_days = len({ts[:10] for ts in user_dates})

    largest_commit_share: float | None = None
    raw_largest_commit_share: float | None = None
    if user_commits:
        sampled = _sample_evenly(user_commits, MAX_SAMPLED_COMMITS)
        raw_counts: list[int] = []
        filtered_counts: list[int] = []
        for commit in sampled:
            sha = commit.get("sha")
            if not sha:
                continue
            detail_resp = _request_with_rate_limit(
                client,
                f"{GITHUB_API_BASE}/repos/{full_name}/commits/{sha}",
            )
            if detail_resp.status_code != 200:
                continue
            detail = detail_resp.json()
            stats = detail.get("stats") or {}
            raw_total = stats.get("total")
            files = detail.get("files") or []
            if not isinstance(raw_total, int):
                raw_total = sum(int(f.get("changes") or 0) for f in files)
            raw_counts.append(raw_total)

            filtered_total = sum(
                int(f.get("changes") or 0)
                for f in files
                if not _is_excluded_from_line_stats(f, file_sizes)
            )
            filtered_counts.append(filtered_total)

        total_raw = sum(raw_counts)
        if raw_counts and total_raw > 0:
            raw_largest_commit_share = round(max(raw_counts) / total_raw, 4)

        total_filtered = sum(filtered_counts)
        if filtered_counts and total_filtered > 0:
            largest_commit_share = round(max(filtered_counts) / total_filtered, 4)

    return {
        "user_commit_count": user_commit_count,
        "unlinked_commit_count": unlinked_commit_count,
        "unlinked_identities": unlinked_identities,
        "other_author_count": other_author_count,
        "confirmed_email_commit_count": confirmed_email_commit_count,
        "total_commit_count": total_commit_count,
        "first_commit_at": first_commit_at,
        "last_commit_at": last_commit_at,
        "active_days": active_days,
        "largest_commit_share": largest_commit_share,
        "raw_largest_commit_share": raw_largest_commit_share,
    }


def _fetch_file_tree(
    client: httpx.Client,
    full_name: str,
    default_branch: str | None,
) -> tuple[list[str], list[str], dict[str, int], bool, int]:
    if not default_branch:
        return [], [], {}, False, 0
    url = f"{GITHUB_API_BASE}/repos/{full_name}/git/trees/{default_branch}"
    response = _request_with_rate_limit(client, url, params={"recursive": 1})
    if response.status_code in (404, 409):
        return [], [], {}, False, 0
    response.raise_for_status()
    data = response.json()
    tree = data.get("tree")
    if not isinstance(tree, list):
        return [], [], {}, False, 0

    paths: list[str] = []
    file_sizes: dict[str, int] = {}
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path = entry.get("path")
        if not path:
            continue
        parts = PurePosixPath(path).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        paths.append(path)
        size = entry.get("size")
        if isinstance(size, int):
            file_sizes[path] = size

    prioritized = sorted(paths, key=lambda p: (p.count("/"), p))
    total_file_count = len(paths)
    truncated = bool(data.get("truncated", False)) or total_file_count > MAX_TREE_PATHS
    return paths, prioritized[:MAX_TREE_PATHS], file_sizes, truncated, total_file_count


def _fetch_readme_excerpt(client: httpx.Client, full_name: str) -> str | None:
    url = f"{GITHUB_API_BASE}/repos/{full_name}/readme"
    response = _request_with_rate_limit(
        client,
        url,
        accept="application/vnd.github.raw+json",
    )
    if response.status_code in (404, 409):
        return None
    if response.status_code != 200:
        return None
    text = response.text.strip()
    if not text:
        return None
    return text[:MAX_README_CHARS]


def _fetch_languages(client: httpx.Client, full_name: str) -> dict[str, int]:
    url = f"{GITHUB_API_BASE}/repos/{full_name}/languages"
    response = _request_with_rate_limit(client, url)
    if response.status_code in (404, 409):
        return {}
    response.raise_for_status()
    data = response.json()
    return {str(k): int(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _fetch_fork_metrics(
    client: httpx.Client,
    repo_data: dict,
    username: str,
) -> tuple[str | None, int | None, int]:
    parent = repo_data.get("parent") or {}
    parent_full_name = parent.get("full_name")
    parent_branch = parent.get("default_branch")
    owner_login = (repo_data.get("owner") or {}).get("login")
    branch = repo_data.get("default_branch")

    ahead_by: int | None = None
    if parent_full_name and parent_branch and owner_login and branch:
        compare_url = f"{GITHUB_API_BASE}/repos/{parent_full_name}/compare/{parent_branch}...{owner_login}:{branch}"
        resp = _request_with_rate_limit(client, compare_url)
        if resp.status_code == 200:
            ahead_by = resp.json().get("ahead_by")

    merged_prs = 0
    if parent_full_name:
        search_url = f"{GITHUB_API_BASE}/search/issues"
        resp = _request_with_rate_limit(
            client,
            search_url,
            params={"q": f"repo:{parent_full_name} type:pr author:{username} is:merged"},
        )
        if resp.status_code == 200:
            merged_prs = int(resp.json().get("total_count", 0))

    return parent_full_name, ahead_by, merged_prs


def _build_repo_evidence(
    client: httpx.Client,
    repo_data: dict,
    username: str,
    confirmed_emails: set[str],
    source: Literal["resume_link", "recent_list"],
) -> RepoEvidence:
    full_name = repo_data["full_name"]
    is_fork = bool(repo_data.get("fork", False))

    if is_fork and "parent" not in repo_data:
        detail_resp = _request_with_rate_limit(client, f"{GITHUB_API_BASE}/repos/{full_name}")
        if detail_resp.status_code == 200:
            repo_data = detail_resp.json()

    size = int(repo_data.get("size", 0) or 0)
    default_branch = repo_data.get("default_branch")

    languages = _fetch_languages(client, full_name)
    readme_excerpt = _fetch_readme_excerpt(client, full_name)

    if size == 0:
        file_tree: list[str] = []
        file_tree_truncated = False
        total_file_count = 0
        dependencies: list[str] = []
        commit_stats = {
            "user_commit_count": 0,
            "unlinked_commit_count": 0,
            "unlinked_identities": [],
            "other_author_count": 0,
            "confirmed_email_commit_count": 0,
            "total_commit_count": 0,
            "first_commit_at": None,
            "last_commit_at": None,
            "active_days": 0,
            "largest_commit_share": None,
            "raw_largest_commit_share": None,
        }
    else:
        all_paths, file_tree, file_sizes, file_tree_truncated, total_file_count = _fetch_file_tree(
            client,
            full_name,
            default_branch,
        )
        dependencies = _extract_dependencies(client, full_name, all_paths)
        commit_stats = _analyze_commits(
            client,
            full_name,
            username,
            confirmed_emails,
            file_sizes,
        )

    parent_full_name: str | None = None
    commits_ahead: int | None = None
    merged_prs: int | None = None
    if is_fork:
        parent_full_name, commits_ahead, merged_prs = _fetch_fork_metrics(
            client,
            repo_data,
            username,
        )

    homepage = repo_data.get("homepage")
    if isinstance(homepage, str) and not homepage.strip():
        homepage = None

    return RepoEvidence(
        name=repo_data["name"],
        full_name=full_name,
        url=repo_data["html_url"],
        description=repo_data.get("description"),
        homepage=homepage,
        is_fork=is_fork,
        parent_full_name=parent_full_name,
        commits_ahead_of_parent=commits_ahead,
        languages=languages,
        topics=repo_data.get("topics") or [],
        stars=int(repo_data.get("stargazers_count", 0) or 0),
        readme_excerpt=readme_excerpt,
        file_tree=file_tree,
        file_tree_truncated=file_tree_truncated,
        total_file_count=total_file_count,
        dependencies=dependencies,
        user_commit_count=commit_stats["user_commit_count"],
        unlinked_commit_count=commit_stats["unlinked_commit_count"],
        unlinked_identities=commit_stats["unlinked_identities"],
        other_author_count=commit_stats["other_author_count"],
        confirmed_email_commit_count=commit_stats["confirmed_email_commit_count"],
        total_commit_count=commit_stats["total_commit_count"],
        first_commit_at=commit_stats["first_commit_at"],
        last_commit_at=commit_stats["last_commit_at"],
        active_days=commit_stats["active_days"],
        largest_commit_share=commit_stats["largest_commit_share"],
        raw_largest_commit_share=commit_stats["raw_largest_commit_share"],
        merged_prs_to_parent=merged_prs,
        source=source,
        fetch_error=None,
    )


def analyze_github(
    username: str,
    resume_repo_links: list[str] | None = None,
    confirmed_emails: list[str] | None = None,
) -> GitHubProfile:
    clean_username = username.strip()
    if not clean_username:
        raise ValueError("GitHub username is empty")

    links = resume_repo_links or []
    confirmed_set = {e.strip().lower() for e in (confirmed_emails or []) if e and e.strip()}

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        user_resp = _request_with_rate_limit(client, f"{GITHUB_API_BASE}/users/{clean_username}")
        if user_resp.status_code == 404:
            raise ValueError(f"GitHub user not found: {clean_username}")
        user_resp.raise_for_status()

        repos: list[RepoEvidence] = []
        seen_full_names: set[str] = set()

        for raw_link in links:
            parsed = _parse_github_repo_link(raw_link)
            if not parsed:
                continue
            owner, repo_name = parsed
            requested_full = f"{owner}/{repo_name}"
            if requested_full.lower() in seen_full_names:
                continue

            repo_resp = _request_with_rate_limit(client, f"{GITHUB_API_BASE}/repos/{requested_full}")
            if repo_resp.status_code == 404:
                seen_full_names.add(requested_full.lower())
                repos.append(
                    RepoEvidence(
                        name=repo_name,
                        full_name=requested_full,
                        url=f"https://github.com/{requested_full}",
                        source="resume_link",
                        fetch_error="Repository not found or private (404)",
                    )
                )
                continue
            repo_resp.raise_for_status()
            repo_data = repo_resp.json()
            actual_full = repo_data["full_name"]
            if actual_full.lower() in seen_full_names:
                continue

            seen_full_names.add(requested_full.lower())
            seen_full_names.add(actual_full.lower())
            repos.append(
                _build_repo_evidence(
                    client,
                    repo_data,
                    clean_username,
                    confirmed_set,
                    source="resume_link",
                )
            )

        if len(repos) < MAX_REPOS:
            list_resp = _request_with_rate_limit(
                client,
                f"{GITHUB_API_BASE}/users/{clean_username}/repos",
                params={"sort": "updated", "direction": "desc", "per_page": 30, "type": "owner"},
            )
            list_resp.raise_for_status()
            recent_repos = list_resp.json()
            if isinstance(recent_repos, list):
                for repo_data in recent_repos:
                    if len(repos) >= MAX_REPOS:
                        break
                    full_name = repo_data.get("full_name")
                    if not full_name or full_name.lower() in seen_full_names:
                        continue
                    seen_full_names.add(full_name.lower())
                    repos.append(
                        _build_repo_evidence(
                            client,
                            repo_data,
                            clean_username,
                            confirmed_set,
                            source="recent_list",
                        )
                    )

    return GitHubProfile(
        username=clean_username,
        repos=repos,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
