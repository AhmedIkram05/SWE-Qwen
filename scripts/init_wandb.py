#!/usr/bin/env python3
"""
W&B Project Initialization Script for SWE-Qwen

This script initializes a Weights & Biases project with proper
artifact tracking, run configuration, and team settings.
"""

import argparse
import os
import sys

import wandb


def init_wandb_project(
    project_name: str = "swe-qwen",
    entity: str = None,
    description: str = None,
    tags: list = None,
    config: dict = None,
):
    """
    Initialize W&B project with standard configuration.

    Args:
        project_name: W&B project name
        entity: W&B entity (username or team name)
        description: Project description
        tags: List of tags for the project
        config: Default configuration for runs
    """
    # Login check
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("ERROR: WANDB_API_KEY not set in environment")
        print("Set it with: export WANDB_API_KEY=your_key")
        print("Or run: wandb login")
        return False

    try:
        wandb.login(key=api_key)
        print("Logged in to W&B successfully")
    except Exception as e:
        print(f"ERROR: Failed to login to W&B: {e}")
        return False

    # Initialize a run to create the project
    run = wandb.init(
        project=project_name,
        entity=entity,
        name="project-init",
        job_type="setup",
        config=config or {},
        tags=tags or ["setup", "initialization"],
        reinit=True,
    )

    # Set project description
    if description:
        run.notes = description

    # Log project metadata
    run.config.update(
        {
            "project": project_name,
            "framework": "pytorch",
            "model_family": "qwen",
            "task": "code-generation",
        }
    )

    # Create default artifacts collection
    artifact = wandb.Artifact(
        name="project-config",
        type="config",
        description="SWE-Qwen project configuration",
        metadata={
            "project": project_name,
            "version": "1.0.0",
        },
    )
    run.log_artifact(artifact)

    # Finish the init run
    run.finish()

    print(f"Project '{project_name}' initialized successfully!")
    print(f"View at: https://wandb.ai/{entity or 'your-username'}/{project_name}")
    return True


def create_sweep_config(project_name: str, entity: str = None):
    """Create a hyperparameter sweep configuration."""
    sweep_config = {
        "name": "swe-qwen-lora-sweep",
        "method": "bayes",
        "metric": {
            "name": "eval_loss",
            "goal": "minimize",
        },
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform",
                "min": -10.819,  # 2e-5
                "max": -8.517,  # 2e-4
            },
            "lora_rank": {
                "values": [32, 64, 128],
            },
            "lora_alpha": {
                "values": [64, 128, 256],
            },
            "lora_dropout": {
                "values": [0.05, 0.1, 0.15],
            },
            "batch_size": {
                "values": [2, 4, 8],
            },
            "num_epochs": {
                "values": [2, 3, 4],
            },
            "warmup_ratio": {
                "distribution": "uniform",
                "min": 0.01,
                "max": 0.1,
            },
            "weight_decay": {
                "distribution": "log_uniform",
                "min": -6.907,  # 0.001
                "max": -2.302,  # 0.1
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
            "eta": 2,
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
    print(f"Sweep created: {sweep_id}")
    print(f"Run with: wandb agent {entity}/{project_name}/{sweep_id}")
    return sweep_id


def setup_artifact_registries(project_name: str, entity: str = None):
    """Set up model and dataset artifact registries."""
    api = wandb.Api()

    # Create model registry
    try:
        api.artifact_type("model", project=project_name, entity=entity)
        print(f"Model registry ready: {project_name}")
    except Exception:
        print("Model registry will be created on first model upload")

    # Create dataset registry
    try:
        api.artifact_type("dataset", project=project_name, entity=entity)
        print(f"Dataset registry ready: {project_name}")
    except Exception:
        print("Dataset registry will be created on first dataset upload")


def main():
    parser = argparse.ArgumentParser(description="Initialize W&B project for SWE-Qwen")
    parser.add_argument("--project", default="swe-qwen", help="W&B project name")
    parser.add_argument("--entity", help="W&B entity (username or team)")
    parser.add_argument("--description", help="Project description")
    parser.add_argument("--create-sweep", action="store_true", help="Create hyperparameter sweep")
    parser.add_argument("--setup-registries", action="store_true", help="Setup artifact registries")
    args = parser.parse_args()

    description = args.description or (
        "SWE-Qwen: Fine-tuning Qwen3-30B-A3B for Software Engineering tasks. "
        "Training on code generation, bug fixing, and repository understanding. "
        "Fallback model: Qwen3-14B."
    )

    tags = ["swe", "qwen", "qwen3-moe", "code-generation", "fine-tuning", "llm"]

    config = {
        "model": "Qwen/Qwen3-30B-A3B",
        "fallback_model": "Qwen/Qwen3-14B",
        "method": "LoRA",
        "quantization": "4-bit",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }

    success = init_wandb_project(
        project_name=args.project,
        entity=args.entity,
        description=description,
        tags=tags,
        config=config,
    )

    if not success:
        return 1

    if args.create_sweep:
        create_sweep_config(args.project, args.entity)

    if args.setup_registries:
        setup_artifact_registries(args.project, args.entity)

    return 0


if __name__ == "__main__":
    sys.exit(main())
