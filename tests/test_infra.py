#!/usr/bin/env python
"""
Tests for Terraform infrastructure outputs.
These tests validate that Terraform modules produce expected outputs
and that the infrastructure graph is correctly configured.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
TERRAFORM_DIR = PROJECT_ROOT / "infra" / "terraform"

# These tests need real GCP credentials because terraform output/plan/graph
# require the GCS backend to be initialized
HAS_GCP_CREDS = bool(
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GCP_SA_KEY")
)


def run_terraform_init(terraform_dir: Path) -> subprocess.CompletedProcess:
    """Run terraform init with local backend for testing."""
    return subprocess.run(
        ["terraform", "init", "-backend=false"],
        cwd=terraform_dir,
        capture_output=True,
        text=True,
        check=False,
    )


def get_terraform_outputs(variables: dict | None = None) -> dict:
    """Get all terraform outputs as a dictionary."""
    terraform_dir = TERRAFORM_DIR

    # First ensure terraform is initialized
    init_result = run_terraform_init(terraform_dir)
    if init_result.returncode != 0:
        raise RuntimeError(f"terraform init failed: {init_result.stderr}")

    cmd = ["terraform", "output", "-json"]
    env = {}
    if variables:
        for k, v in variables.items():
            # Convert bool to string for Terraform
            if isinstance(v, bool):
                env[f"TF_VAR_{k}"] = "true" if v else "false"
            else:
                env[f"TF_VAR_{k}"] = str(v)

    result = subprocess.run(
        cmd,
        cwd=terraform_dir,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"terraform output failed: {result.stderr}")
    return json.loads(result.stdout)


@pytest.fixture(scope="session")
def terraform_initialized():
    """Initialize Terraform once per test session."""
    result = subprocess.run(
        ["terraform", "init", "-backend=false"],
        cwd=TERRAFORM_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"terraform init failed: {result.stderr}"
    return True


@pytest.mark.skipif(
    shutil.which("terraform") is None,
    reason="terraform binary not available on this runner",
)
class TestTerraformInit:
    """Test Terraform initialization and validation."""

    def test_terraform_init(self, terraform_initialized):
        """Terraform should initialize without errors."""
        assert terraform_initialized

    def test_terraform_validate(self, terraform_initialized):
        """Terraform configuration should be valid."""
        result = subprocess.run(
            ["terraform", "validate"],
            cwd=TERRAFORM_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"terraform validate failed: {result.stderr}"

    def test_terraform_fmt_check(self):
        """Terraform files should be formatted."""
        result = subprocess.run(
            ["terraform", "fmt", "-check", "-diff"],
            cwd=TERRAFORM_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"terraform fmt check failed: {result.stdout}"


@pytest.mark.skipif(not HAS_GCP_CREDS, reason="Requires GCP credentials")
class TestStorageModuleOutputs:
    """Test storage module outputs."""

    @pytest.fixture(scope="class")
    def tf_vars(self):
        return {
            "gcp_project_id": "test-project",
            "gcp_region": "us-central1",
            "environment": "dev",
            "repository_name": "SWE-Qwen",
            "repository_owner": "test-owner",
        }

    def test_storage_module_has_dataset_bucket_output(self, tf_vars):
        """Storage module should output dataset_bucket_name."""
        outputs = get_terraform_outputs(tf_vars)
        assert "dataset_bucket_name" in outputs
        assert outputs["dataset_bucket_name"]["value"]

    def test_storage_module_has_model_bucket_output(self, tf_vars):
        """Storage module should output model_bucket_name."""
        outputs = get_terraform_outputs(tf_vars)
        assert "model_bucket_name" in outputs
        assert outputs["model_bucket_name"]["value"]

    def test_storage_module_has_bucket_locations(self, tf_vars):
        """Storage module should output bucket locations."""
        outputs = get_terraform_outputs(tf_vars)
        assert "dataset_bucket_location" in outputs
        assert "model_bucket_location" in outputs
        assert outputs["dataset_bucket_location"]["value"] == "US-CENTRAL1"
        assert outputs["model_bucket_location"]["value"] == "US-CENTRAL1"


@pytest.mark.skipif(not HAS_GCP_CREDS, reason="Requires GCP credentials")
class TestIAMModuleOutputs:
    """Test IAM module outputs."""

    @pytest.fixture(scope="class")
    def tf_vars(self):
        return {
            "gcp_project_id": "test-project",
            "gcp_region": "us-central1",
            "environment": "dev",
            "repository_name": "SWE-Qwen",
            "repository_owner": "test-owner",
            "enable_workload_identity": True,
        }

    def test_iam_module_has_modal_runner_sa(self, tf_vars):
        """IAM module should output modal_runner_service_account_email."""
        outputs = get_terraform_outputs(tf_vars)
        assert "modal_runner_service_account_email" in outputs
        sa_email = outputs["modal_runner_service_account_email"]["value"]
        assert sa_email.endswith("@test-project.iam.gserviceaccount.com")

    def test_iam_module_has_github_actions_sa(self, tf_vars):
        """IAM module should output github_actions_service_account_email when WIF enabled."""
        outputs = get_terraform_outputs(tf_vars)
        assert "github_actions_service_account_email" in outputs
        sa_email = outputs["github_actions_service_account_email"]["value"]
        assert sa_email.endswith("@test-project.iam.gserviceaccount.com")

    def test_iam_module_has_cloud_build_sa(self, tf_vars):
        """IAM module should output cloud_build_service_account_email."""
        outputs = get_terraform_outputs(tf_vars)
        assert "cloud_build_service_account_email" in outputs
        sa_email = outputs["cloud_build_service_account_email"]["value"]
        assert sa_email.endswith("@test-project.iam.gserviceaccount.com")

    def test_iam_module_has_wif_pool(self, tf_vars):
        """IAM module should output workload_identity_pool_name when WIF enabled."""
        outputs = get_terraform_outputs(tf_vars)
        assert "workload_identity_pool_name" in outputs
        pool_name = outputs["workload_identity_pool_name"]["value"]
        assert "github-actions-pool-dev" in pool_name

    def test_iam_module_has_wif_provider(self, tf_vars):
        """IAM module should output workload_identity_pool_provider_name."""
        outputs = get_terraform_outputs(tf_vars)
        assert "workload_identity_pool_provider_name" in outputs
        provider_name = outputs["workload_identity_pool_provider_name"]["value"]
        assert "github-provider-dev" in provider_name

    def test_iam_module_has_secret_names(self, tf_vars):
        """IAM module should output secret names."""
        outputs = get_terraform_outputs(tf_vars)
        assert "modal_token_secret_name" in outputs
        assert "wandb_api_key_secret_name" in outputs
        assert "github_token_secret_name" in outputs
        assert outputs["modal_token_secret_name"]["value"] == "modal-token"
        assert outputs["wandb_api_key_secret_name"]["value"] == "wandb-api-key"
        assert outputs["github_token_secret_name"]["value"] == "github-token"

    def test_iam_module_has_artifact_registry(self, tf_vars):
        """IAM module should output artifact registry repository."""
        outputs = get_terraform_outputs(tf_vars)
        assert "artifact_registry_repository" in outputs
        assert "artifact_registry_location" in outputs
        repo = outputs["artifact_registry_repository"]["value"]
        assert "swe-qwen-dev" in repo


@pytest.mark.skipif(not HAS_GCP_CREDS, reason="Requires GCP credentials")
class TestModuleDependencies:
    """Test that modules correctly reference each other."""

    @pytest.fixture(scope="class")
    def tf_vars(self):
        return {
            "gcp_project_id": "test-project",
            "gcp_region": "us-central1",
            "environment": "dev",
            "repository_name": "SWE-Qwen",
            "repository_owner": "test-owner",
            "enable_workload_identity": True,
        }

    def test_iam_module_receives_storage_outputs(self, tf_vars):
        """IAM module should receive bucket names from storage module."""
        outputs = get_terraform_outputs(tf_vars)
        # IAM module uses these for bucket IAM bindings
        # The test validates the module composition works
        assert "dataset_bucket_name" in outputs
        assert "model_bucket_name" in outputs


@pytest.mark.skipif(not HAS_GCP_CREDS, reason="Requires GCP credentials")
class TestInfrastructureGraph:
    """Test the complete infrastructure dependency graph."""

    def test_no_circular_dependencies(self):
        """Validate no circular dependencies in Terraform graph."""
        result = subprocess.run(
            ["terraform", "graph"],
            cwd=TERRAFORM_DIR,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "TF_VAR_gcp_project_id": "test-project",
                "TF_VAR_gcp_region": "us-central1",
                "TF_VAR_environment": "dev",
                "TF_VAR_repository_name": "SWE-Qwen",
                "TF_VAR_repository_owner": "test-owner",
                "TF_VAR_enable_workload_identity": "true",
            },
            check=False,
        )
        assert result.returncode == 0, f"terraform graph failed: {result.stderr}"

        # Check for cycles in the graph (simplified check)
        graph_output = result.stdout
        assert "digraph" in graph_output
        # No explicit cycle detection - Terraform would fail on cycles

    def test_required_resources_present(self):
        """Key resources should be present in the graph."""
        result = subprocess.run(
            ["terraform", "graph"],
            cwd=TERRAFORM_DIR,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "TF_VAR_gcp_project_id": "test-project",
                "TF_VAR_gcp_region": "us-central1",
                "TF_VAR_environment": "dev",
                "TF_VAR_repository_name": "SWE-Qwen",
                "TF_VAR_repository_owner": "test-owner",
                "TF_VAR_enable_workload_identity": "true",
            },
            check=False,
        )
        assert result.returncode == 0

        graph = result.stdout
        required_resources = [
            "google_storage_bucket.dataset",
            "google_storage_bucket.model",
            "google_service_account.modal_runner",
            "google_service_account.github_actions",
            "google_service_account.cloud_build",
            "google_iam_workload_identity_pool.github_pool",
            "google_iam_workload_identity_pool_provider.github_provider",
            "google_secret_manager_secret.modal_token",
            "google_secret_manager_secret.wandb_api_key",
            "google_secret_manager_secret.github_token",
            "google_artifact_registry_repository.docker_repo",
        ]

        for resource in required_resources:
            assert resource in graph, f"Missing resource in graph: {resource}"


@pytest.mark.integration
class TestTerraformPlan:
    """Test terraform plan execution (requires GCP credentials)."""

    def test_terraform_plan_dry_run(self):
        """Terraform plan should execute without errors (dry run)."""
        result = subprocess.run(
            [
                "terraform",
                "plan",
                "-var=gcp_project_id=test-project",
                "-var=gcp_region=us-central1",
                "-var=environment=dev",
                "-var=repository_name=SWE-Qwen",
                "-var=repository_owner=test-owner",
                "-var=enable_workload_identity=true",
                "-detailed-exitcode",
            ],
            cwd=TERRAFORM_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        # Exit code 0 = no changes, 2 = changes pending, both are valid
        # Exit code 1 = error
        assert result.returncode != 1, f"terraform plan failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
