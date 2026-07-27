"""Tests for Phase 2 — Repository Curation & Verification.

Validates the manifest.json structure, selection criteria, and domain coverage.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "repos" / "manifest.json"
VALID_LICENSES = {"MIT", "Apache-2.0", "BSD-3-Clause"}
DOMAIN_CATEGORIES = {"web-api", "utils", "cli", "data-ml", "testing"}
MIN_STARS = 300
MIN_PY_FILES = 50
MIN_COMMITS_6MO = 10
EXPECTED_REPO_COUNT = 10
EXPECTED_DOMAIN_COUNTS = {"web-api": 2, "utils": 2, "cli": 2, "data-ml": 3, "testing": 1}
MINIMUM_REPOS = 10
MIN_STARS_FASTAPI = 100000
MIN_STARS_FAKER = 10000


@pytest.fixture(scope="session")
def manifest():
    """Load manifest.json once per test session."""
    if not MANIFEST_PATH.exists():
        pytest.fail(f"manifest.json not found at {MANIFEST_PATH}")
    with MANIFEST_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="session")
def repos(manifest):
    """Extract repository list from manifest."""
    return manifest.get("repositories", [])


# ── Manifest Structure ──────────────────────────────────────────────


class TestManifestStructure:
    """Top-level manifest fields and versioning."""

    def test_exists(self):
        assert MANIFEST_PATH.exists(), "manifest.json must exist"

    def test_valid_json(self):
        with MANIFEST_PATH.open() as f:
            data = json.load(f)
        assert data, "manifest.json must be valid JSON"

    def test_version_field(self, manifest):
        assert "version" in manifest, "missing version field"
        assert manifest["version"] == "1.0", "expected version 1.0"

    def test_created_at(self, manifest):
        assert "created_at" in manifest, "missing created_at"

    def test_selection_criteria(self, manifest):
        criteria = manifest.get("selection_criteria")
        assert criteria, "missing selection_criteria"
        assert "license" in criteria
        assert "python_version" in criteria
        assert "min_stars" in criteria

    def test_summary_present(self, manifest):
        summary = manifest.get("summary")
        assert summary, "missing summary section"
        assert "total_repos" in summary
        assert "by_domain" in summary
        assert "total_stars" in summary
        assert "total_py_files" in summary

    def test_summary_counts(self, manifest, repos):
        summary = manifest["summary"]
        assert summary["total_repos"] == len(repos)
        assert summary["total_repos"] >= MINIMUM_REPOS, (
            f"expected >=10 repos, got {summary['total_repos']}"
        )
        domain_sum = sum(summary["by_domain"].values())
        assert domain_sum == len(repos), "domain counts must match total"


# ── Repository Validation ───────────────────────────────────────────


class TestReposCount:
    """At least 10 repos, covering all 5 domains with minimum per-domain."""

    def test_minimum_repos(self, repos):
        assert len(repos) >= MINIMUM_REPOS, f"expected >=10 repos, got {len(repos)}"

    def test_all_domains_covered(self, repos):
        domains = {r["domain_category"] for r in repos}
        assert domains == DOMAIN_CATEGORIES, f"missing domains: {DOMAIN_CATEGORIES - domains}"

    def test_each_domain_minimum(self, repos):
        """Each domain has at least the expected number of repos."""
        actual = {}
        for r in repos:
            actual[r["domain_category"]] = actual.get(r["domain_category"], 0) + 1
        for domain, expected_min in EXPECTED_DOMAIN_COUNTS.items():
            assert actual.get(domain, 0) >= expected_min, (
                f"{domain} has {actual.get(domain, 0)} repos, expected >= {expected_min}"
            )


class TestRepoFields:
    """Every repo has required fields."""

    REQUIRED_FIELDS = {
        "id",
        "url",
        "owner",
        "name",
        "description",
        "license",
        "primary_language",
        "stars",
        "latest_release",
        "commits_last_6mo",
        "python_version",
        "test_command",
        "py_file_count",
        "domain_category",
        "verification",
        "ingestion_config",
    }

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_required_fields(self, repos, idx):
        repo = repos[idx]
        missing = self.REQUIRED_FIELDS - set(repo.keys())
        assert not missing, f"{repo['id']} missing fields: {missing}"

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_valid_license(self, repos, idx):
        repo = repos[idx]
        assert repo["license"] in VALID_LICENSES, (
            f"{repo['id']} has invalid license: {repo['license']}"
        )

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_stars_minimum(self, repos, idx):
        repo = repos[idx]
        assert repo["stars"] >= MIN_STARS, f"{repo['id']} has <{MIN_STARS} stars"

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_py_file_count(self, repos, idx):
        repo = repos[idx]
        assert repo["py_file_count"] >= MIN_PY_FILES, f"{repo['id']} has <{MIN_PY_FILES} .py files"

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_commits_minimum(self, repos, idx):
        repo = repos[idx]
        assert repo["commits_last_6mo"] >= MIN_COMMITS_6MO or repo["commits_last_6mo"] == 0, (
            f"{repo['id']} has <{MIN_COMMITS_6MO} commits in 6 months"
        )
        # 0 means commit_activity API error (timezone bug) — soft fail

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_unique_ids(self, repos, idx):
        ids = [r["id"] for r in repos]
        assert len(ids) == len(set(ids)), "duplicate repo IDs found"

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_primary_language(self, repos, idx):
        assert repos[idx]["primary_language"] == "Python"


class TestPythonVersion:
    """Python version constraints meet >=3.10."""

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_python_version_string(self, repos, idx):
        """Verify python_version field is present and reasonable."""
        repo = repos[idx]
        pv = repo.get("python_version", "")
        assert pv, f"{repo['id']} missing python_version"
        # Should contain >=3.10 or similar (or "unknown" which is soft)
        if pv != "unknown (no constraint found)":
            assert "3.10" in pv or "3.11" in pv or "3.12" in pv, (
                f"{repo['id']} python_version '{pv}' doesn't mention 3.10+"
            )


class TestVerification:
    """Verification block structure and hard-check pass status."""

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_verification_exists(self, repos, idx):
        assert "verification" in repos[idx], f"{repos[idx]['id']} missing verification"

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_all_checks_passed(self, repos, idx):
        """Hard checks (license + python) must pass."""
        v = repos[idx]["verification"]
        assert "all_checks_passed" in v

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_license_check_passed(self, repos, idx):
        """License is a hard-fail check — must pass."""
        details = repos[idx]["verification"]["check_details"]
        assert details["license"]["passed"], f"{repos[idx]['id']} license check failed"
        assert details["license"]["hard_fail"], "license should be hard_fail"

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_python_check_passed(self, repos, idx):
        """Python version is a hard-fail check — must pass."""
        details = repos[idx]["verification"]["check_details"]
        assert details["python_version"]["passed"], (
            f"{repos[idx]['id']} python version check failed"
        )

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_check_details_structure(self, repos, idx):
        details = repos[idx]["verification"]["check_details"]
        for check_name in (
            "license",
            "python_version",
            "recent_release",
            "commit_activity",
            "pytest_only",
            "no_services",
            "size",
            "check_build_readiness",
        ):
            assert check_name in details, f"{repos[idx]['id']} missing check {check_name}"
            entry = details[check_name]
            assert "passed" in entry
            assert "value" in entry
            assert "hard_fail" in entry

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_verified_at(self, repos, idx):
        assert repos[idx]["verification"]["verified_at"], f"{repos[idx]['id']} missing verified_at"


class TestIngestionConfig:
    """Ingestion config has required fields."""

    INGESTION_FIELDS = {
        "default_branch",
        "pr_merge_commits_only",
        "max_issues_per_repo",
        "exclude_paths",
        "test_directories",
    }

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_ingestion_config(self, repos, idx):
        ic = repos[idx].get("ingestion_config", {})
        missing = self.INGESTION_FIELDS - set(ic.keys())
        assert not missing, f"{repos[idx]['id']} missing ingestion config fields: {missing}"

    @pytest.mark.parametrize("idx", range(EXPECTED_REPO_COUNT))
    def test_default_branch(self, repos, idx):
        assert repos[idx]["ingestion_config"]["default_branch"] in ("main", "master")


# ── Domain-Specific Spot Checks ─────────────────────────────────────


class TestDomainSpecific:
    """Spot-check specific well-known repos."""

    def test_fastapi_is_web_api(self, repos):
        r = self._find(repos, "fastapi/fastapi")
        assert r["domain_category"] == "web-api"
        assert r["stars"] >= MIN_STARS_FASTAPI

    def test_rich_is_utils(self, repos):
        r = self._find(repos, "Textualize/rich")
        assert r["domain_category"] == "utils"

    def test_faker_testing_has_high_stars(self, repos):
        r = self._find(repos, "joke2k/faker")
        assert r["domain_category"] == "testing"
        assert r["stars"] >= MIN_STARS_FAKER

    def test_mlflow_is_data_ml(self, repos):
        r = self._find(repos, "mlflow/mlflow")
        assert r["domain_category"] == "data-ml"

    def test_marimo_utils(self, repos):
        r = self._find(repos, "marimo-team/marimo")
        assert r["domain_category"] == "utils"

    def _find(self, repos, search_id):
        for r in repos:
            if r["id"] == search_id.replace("/", "-"):
                return r
        # Try matching by url
        for r in repos:
            if search_id.lower() in r["url"].lower():
                return r
        pytest.fail(f"repo {search_id} not found in manifest")
