# Main Terraform Configuration for SWE-Qwen Platform
#
# This configuration provisions:
# - GCS buckets for datasets and model checkpoints
# - IAM roles and Workload Identity Federation for GitHub Actions
# - Secret Manager for API keys
# - Service accounts for Modal and CI/CD

module "storage" {
  source = "./modules/storage"

  project_id     = var.gcp_project_id
  region         = var.gcp_region
  environment    = var.environment
  dataset_bucket = var.dataset_bucket_name
  model_bucket   = var.model_bucket_name
  labels         = var.labels
}

module "iam" {
  source = "./modules/iam"

  project_id               = var.gcp_project_id
  region                   = var.gcp_region
  environment              = var.environment
  repository_name          = var.repository_name
  repository_owner         = var.repository_owner
  enable_workload_identity = var.enable_workload_identity
  modal_token_secret       = var.modal_token_secret_name
  wandb_api_key_secret     = var.wandb_api_key_secret_name
  github_token_secret      = var.github_token_secret_name
  labels                   = var.labels
  # Use static variable values (not module outputs) so count conditions
  # in the IAM module can be evaluated at plan time. The actual bucket
  # names from storage module outputs are used at apply time.
  dataset_bucket_name = var.dataset_bucket_name
  model_bucket_name   = var.model_bucket_name
}

# Random suffix for unique bucket names if not provided
resource "random_string" "bucket_suffix" {
  length  = 6
  special = false
  upper   = false
}

# Root outputs are defined in outputs.tf
