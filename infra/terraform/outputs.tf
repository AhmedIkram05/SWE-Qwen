# Root Terraform Outputs
# Re-exports module outputs for easy access at the root level

output "dataset_bucket_name" {
  description = "Name of the dataset GCS bucket"
  value       = module.storage.dataset_bucket_name
}

output "dataset_bucket_location" {
  description = "Location of the dataset bucket"
  value       = module.storage.dataset_bucket_location
}

output "model_bucket_name" {
  description = "Name of the model GCS bucket"
  value       = module.storage.model_bucket_name
}

output "model_bucket_location" {
  description = "Location of the model checkpoint bucket"
  value       = module.storage.model_bucket_location
}

output "modal_runner_service_account_email" {
  description = "Email of the Modal Runner service account"
  value       = module.iam.modal_runner_service_account_email
}

output "github_actions_service_account_email" {
  description = "Email of the GitHub Actions service account"
  value       = module.iam.github_actions_service_account_email
}

output "cloud_build_service_account_email" {
  description = "Email of the Cloud Build service account"
  value       = module.iam.cloud_build_service_account_email
}

output "workload_identity_pool_name" {
  description = "Name of the WIF pool"
  value       = module.iam.workload_identity_pool_name
}

output "workload_identity_pool_provider_name" {
  description = "Name of the WIF provider"
  value       = module.iam.workload_identity_pool_provider_name
}

output "modal_token_secret_name" {
  description = "Name of the Modal API token secret"
  value       = module.iam.modal_token_secret_name
}

output "wandb_api_key_secret_name" {
  description = "Name of the W&B API key secret"
  value       = module.iam.wandb_api_key_secret_name
}

output "github_token_secret_name" {
  description = "Name of the GitHub token secret"
  value       = module.iam.github_token_secret_name
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository name"
  value       = module.iam.artifact_registry_repository
}

output "artifact_registry_location" {
  description = "Artifact Registry location"
  value       = module.iam.artifact_registry_location
}
