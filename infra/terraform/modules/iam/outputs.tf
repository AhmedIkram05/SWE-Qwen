# IAM Module Outputs

output "modal_runner_service_account_email" {
  description = "Email of the Modal Runner service account"
  value       = google_service_account.modal_runner.email
}

output "modal_runner_service_account_name" {
  description = "Name of the Modal Runner service account"
  value       = google_service_account.modal_runner.name
}

output "github_actions_service_account_email" {
  description = "Email of the GitHub Actions service account"
  value       = var.enable_workload_identity ? google_service_account.github_actions[0].email : ""
}

output "github_actions_service_account_name" {
  description = "Name of the GitHub Actions service account"
  value       = var.enable_workload_identity ? google_service_account.github_actions[0].name : ""
}

output "cloud_build_service_account_email" {
  description = "Email of the Cloud Build service account"
  value       = google_service_account.cloud_build.email
}

output "workload_identity_pool_name" {
  description = "Name of the Workload Identity Pool"
  value       = var.enable_workload_identity ? google_iam_workload_identity_pool.github_pool[0].name : ""
}

output "workload_identity_pool_provider_name" {
  description = "Name of the Workload Identity Pool Provider"
  value       = var.enable_workload_identity ? google_iam_workload_identity_pool_provider.github_provider[0].name : ""
}

output "modal_token_secret_name" {
  description = "Name of the Modal token secret"
  value       = google_secret_manager_secret.modal_token.secret_id
}

output "wandb_api_key_secret_name" {
  description = "Name of the W&B API key secret"
  value       = google_secret_manager_secret.wandb_api_key.secret_id
}

output "github_token_secret_name" {
  description = "Name of the GitHub token secret"
  value       = google_secret_manager_secret.github_token.secret_id
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository name"
  value       = google_artifact_registry_repository.docker_repo.name
}

output "artifact_registry_location" {
  description = "Artifact Registry location"
  value       = google_artifact_registry_repository.docker_repo.location
}
