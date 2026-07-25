# Root Terraform Outputs
# Re-exports module outputs for easy access at the root level

output "dataset_bucket_name" {
  description = "Name of the dataset GCS bucket"
  value       = module.storage.dataset_bucket_name
}

output "model_bucket_name" {
  description = "Name of the model GCS bucket"
  value       = module.storage.model_bucket_name
}

output "modal_runner_service_account_email" {
  description = "Email of the Modal Runner service account"
  value       = module.iam.modal_runner_service_account_email
}

output "github_actions_service_account_email" {
  description = "Email of the GitHub Actions service account"
  value       = module.iam.github_actions_service_account_email
}

output "workload_identity_pool_provider_name" {
  description = "Name of the WIF provider"
  value       = module.iam.workload_identity_pool_provider_name
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository name"
  value       = module.iam.artifact_registry_repository
}
