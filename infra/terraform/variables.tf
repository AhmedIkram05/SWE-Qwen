variable "gcp_project_id" {
  description = "GCP Project ID for infrastructure deployment"
  type        = string
}

variable "gcp_region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "repository_name" {
  description = "GitHub repository name for OIDC federation"
  type        = string
  default     = "SWE-Qwen"
}

variable "repository_owner" {
  description = "GitHub repository owner/organization"
  type        = string
  default     = "AhmedIkram05"
}

variable "dataset_bucket_name" {
  description = "Name of the GCS bucket for dataset storage"
  type        = string
  default     = "swe-qwen-datasets"
}

variable "model_bucket_name" {
  description = "Name of the GCS bucket for model checkpoints"
  type        = string
  default     = ""
}

variable "terraform_state_bucket" {
  description = "GCS bucket for Terraform state (created separately)"
  type        = string
  default     = "swe-qwen-terraform-state"
}

variable "allowed_gcp_domains" {
  description = "Allowed GCP domains for organization policies"
  type        = list(string)
  default     = []
}

variable "enable_workload_identity" {
  description = "Enable Workload Identity Federation for GitHub Actions"
  type        = bool
  default     = true
}

variable "modal_token_secret_name" {
  description = "Secret Manager secret name for Modal API token"
  type        = string
  default     = "modal-api-token"
}

variable "wandb_api_key_secret_name" {
  description = "Secret Manager secret name for W&B API key"
  type        = string
  default     = "wandb-api-key"
}

variable "github_token_secret_name" {
  description = "Secret Manager secret name for GitHub token (legacy — data pipeline now uses SWE-bench + BigQuery)"
  type        = string
  default     = "github-token"
}

variable "labels" {
  description = "Common labels for all resources"
  type        = map(string)
  default = {
    project     = "swe-qwen"
    managed_by  = "terraform"
    environment = "dev"
  }
}
