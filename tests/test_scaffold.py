#!/usr/bin/env python
"""
Tests for project scaffold structure and configuration.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestProjectStructure:
    """Test that all required directories and files exist."""

    def test_pyproject_toml_exists(self):
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_gitignore_exists(self):
        assert (PROJECT_ROOT / ".gitignore").exists()

    def test_readme_exists(self):
        assert (PROJECT_ROOT / "README.md").exists()

    def test_modal_app_exists(self):
        assert (PROJECT_ROOT / "modal_app.py").exists()

    def test_scripts_directory_exists(self):
        assert (PROJECT_ROOT / "scripts").is_dir()

    def test_init_wandb_script_exists(self):
        assert (PROJECT_ROOT / "scripts" / "init_wandb.py").exists()

    def test_github_workflows_directory_exists(self):
        assert (PROJECT_ROOT / ".github" / "workflows").is_dir()

    def test_ci_workflow_exists(self):
        assert (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").exists()

    def test_terraform_directory_exists(self):
        assert (PROJECT_ROOT / "infra" / "terraform").is_dir()

    def test_terraform_main_exists(self):
        assert (PROJECT_ROOT / "infra" / "terraform" / "main.tf").exists()

    def test_terraform_variables_exists(self):
        assert (PROJECT_ROOT / "infra" / "terraform" / "variables.tf").exists()

    def test_terraform_providers_exists(self):
        assert (PROJECT_ROOT / "infra" / "terraform" / "providers.tf").exists()

    def test_storage_module_exists(self):
        assert (PROJECT_ROOT / "infra" / "terraform" / "modules" / "storage").is_dir()
        assert (PROJECT_ROOT / "infra" / "terraform" / "modules" / "storage" / "main.tf").exists()
        assert (PROJECT_ROOT / "infra" / "terraform" / "modules" / "storage" / "variables.tf").exists()
        assert (PROJECT_ROOT / "infra" / "terraform" / "modules" / "storage" / "outputs.tf").exists()

    def test_iam_module_exists(self):
        assert (PROJECT_ROOT / "infra" / "terraform" / "modules" / "iam").is_dir()
        assert (PROJECT_ROOT / "infra" / "terraform" / "modules" / "iam" / "main.tf").exists()
        assert (PROJECT_ROOT / "infra" / "terraform" / "modules" / "iam" / "variables.tf").exists()
        assert (PROJECT_ROOT / "infra" / "terraform" / "modules" / "iam" / "outputs.tf").exists()

    def test_tests_directory_exists(self):
        assert (PROJECT_ROOT / "tests").is_dir()


class TestPyProjectToml:
    """Test pyproject.toml configuration."""

    def test_pyproject_toml_valid_toml(self):
        import tomllib
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

    def test_project_metadata(self):
        import tomllib
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        project = config.get("project", {})
        assert project.get("name") == "swe-qwen"
        assert "version" in project
        assert "description" in project
        assert "authors" in project
        assert "license" in project
        assert "requires-python" in project

    def test_dependencies_defined(self):
        import tomllib
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        project = config.get("project", {})
        deps = project.get("dependencies", [])
        required_deps = [
            "torch",
            "transformers",
            "peft",
            "bitsandbytes",
            "accelerate",
            "trl",
            "datasets",
            "wandb",
            "modal",
            "vllm",
        ]
        for dep in required_deps:
            assert any(dep in d for d in deps), f"Missing dependency: {dep}"

    def test_dev_dependencies_defined(self):
        import tomllib
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        optional_deps = config.get("project", {}).get("optional-dependencies", {})
        dev_deps = optional_deps.get("dev", [])
        required_dev = ["pytest", "ruff", "mypy", "pre-commit"]
        for dep in required_dev:
            assert any(dep in d for d in dev_deps), f"Missing dev dependency: {dep}"

    def test_tool_configurations(self):
        import tomllib
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        tools = config.get("tool", {})
        assert "ruff" in tools
        assert "mypy" in tools
        assert "pytest" in tools
        assert "pre-commit-hooks" in tools


class TestGitIgnore:
    """Test .gitignore contains required patterns."""

    def test_gitignore_contains_model_patterns(self):
        content = (PROJECT_ROOT / ".gitignore").read_text()
        required_patterns = [
            "models/checkpoints/",
            "wandb/",
            ".modal/",
            ".env",
            "*.tfstate",
            "*.tfstate.*",
            ".terraform/",
            "*.pkl",
            "*.bin",
            "*.safetensors",
        ]
        for pattern in required_patterns:
            assert pattern in content, f"Missing pattern in .gitignore: {pattern}"


class TestModalApp:
    """Test modal_app.py structure."""

    def test_modal_app_imports(self):
        """Test that modal_app.py can be parsed."""
        content = (PROJECT_ROOT / "modal_app.py").read_text()
        assert "import modal" in content
        assert "app = modal.App" in content or 'modal.App(' in content

    def test_modal_app_has_functions(self):
        content = (PROJECT_ROOT / "modal_app.py").read_text()
        required_functions = [
            "hello_modal",
            "train_swe_qwen",
            "serve_swe_qwen",
        ]
        for func in required_functions:
            assert f"def {func}" in content or f"@app.function" in content, f"Missing function: {func}"


class TestInitWandbScript:
    """Test init_wandb.py script."""

    def test_script_executable(self):
        script = PROJECT_ROOT / "scripts" / "init_wandb.py"
        assert os.access(script, os.R_OK)

    def test_script_has_main(self):
        content = (PROJECT_ROOT / "scripts" / "init_wandb.py").read_text()
        assert "def main" in content
        assert 'if __name__ == "__main__"' in content


class TestTerraformFiles:
    """Test Terraform files have valid syntax."""

    def test_terraform_init(self):
        """Run terraform init to validate configuration."""
        terraform_dir = PROJECT_ROOT / "infra" / "terraform"
        result = subprocess.run(
            ["terraform", "init", "-backend=false"],
            cwd=terraform_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"terraform init failed: {result.stderr}"

    def test_terraform_validate(self):
        """Run terraform validate to check syntax."""
        terraform_dir = PROJECT_ROOT / "infra" / "terraform"
        result = subprocess.run(
            ["terraform", "validate"],
            cwd=terraform_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"terraform validate failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])