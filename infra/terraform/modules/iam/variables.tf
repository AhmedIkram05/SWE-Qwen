# IAM Module Variables

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "repository_owner" {
  description = "GitHub repository owner (org or user)"
  type        = string
}

variable "repository_name" {
  description = "GitHub repository name"
  type        = string
}

variable "enable_workload_identity" {
  description = "Enable Workload Identity Federation for GitHub Actions"
  type        = bool
  default     = true
}

variable "modal_token_secret" {
  description = "Secret Manager secret name for Modal token"
  type        = string
  default     = "modal-token"
}

variable "wandb_api_key_secret" {
  description = "Secret Manager secret name for W&B API key"
  type        = string
  default     = "wandb-api-key"
}

variable "github_token_secret" {
  description = "Secret Manager secret name for GitHub token"
  type        = string
  default     = "github-token"
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}

variable "storage_admin_emails" {
  description = "List of emails to grant Storage Admin role"
  type        = list(string)
  default     = []
}

variable "dataset_bucket_name" {
  description = "Name of the dataset GCS bucket"
  type        = string
  default     = ""
}

variable "model_bucket_name" {
  description = "Name of the model checkpoint GCS bucket"
  type        = string
  default     = ""
}