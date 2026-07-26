# IAM Module - Main Configuration

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "container.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "storage.googleapis.com"
  ])

  service            = each.value
  disable_on_destroy = false
}

# Service Account for Modal Runner
resource "google_service_account" "modal_runner" {
  account_id   = "modal-runner-${var.environment}"
  display_name = "Modal Runner Service Account (${var.environment})"
  description  = "Service account for Modal GPU runners to access GCS and Secret Manager"
}

# Service Account for GitHub Actions
resource "google_service_account" "github_actions" {
  count        = var.enable_workload_identity ? 1 : 0
  account_id   = "github-actions-${var.environment}"
  display_name = "GitHub Actions Service Account (${var.environment})"
  description  = "Service account for GitHub Actions via Workload Identity Federation"
}

# Service Account for Cloud Build
resource "google_service_account" "cloud_build" {
  account_id   = "cloudbuild-${var.environment}"
  display_name = "Cloud Build Service Account (${var.environment})"
  description  = "Service account for Cloud Build deployments"
}

# Workload Identity Pool for GitHub Actions
resource "google_iam_workload_identity_pool" "github_pool" {
  count                     = var.enable_workload_identity ? 1 : 0
  workload_identity_pool_id = "github-actions-pool-${var.environment}"
  display_name              = "GitHub Actions Pool (${var.environment})"
  description               = "Workload Identity Pool for GitHub Actions"
}

# Workload Identity Pool Provider for GitHub
resource "google_iam_workload_identity_pool_provider" "github_provider" {
  count = var.enable_workload_identity ? 1 : 0

  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool[0].workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider-${var.environment}"
  display_name                       = "GitHub Provider (${var.environment})"
  description                        = "OIDC provider for GitHub Actions"

  attribute_mapping = {
    "google.subject"             = "assertion.sub"
    "attribute.actor"            = "assertion.actor"
    "attribute.repository"       = "assertion.repository"
    "attribute.repository_owner" = "assertion.repository_owner"
    "attribute.ref"              = "assertion.ref"
    "attribute.sha"              = "assertion.sha"
    "attribute.workflow"         = "assertion.workflow"
  }

  attribute_condition = "assertion.repository_owner == '${var.repository_owner}' && assertion.repository == '${var.repository_owner}/${var.repository_name}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Allow GitHub Actions to impersonate the GitHub Actions service account
resource "google_service_account_iam_binding" "github_wif_binding" {
  count = var.enable_workload_identity ? 1 : 0

  service_account_id = google_service_account.github_actions[0].name
  role               = "roles/iam.workloadIdentityUser"
  members = [
    "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool[0].name}/attribute.repository/${var.repository_owner}/${var.repository_name}"
  ]
}

# Secret Manager Secrets
resource "google_secret_manager_secret" "modal_token" {
  secret_id = var.modal_token_secret

  replication {
    auto {}
  }

  labels = merge(var.labels, {
    component   = "modal"
    environment = var.environment
  })
}

resource "google_secret_manager_secret" "wandb_api_key" {
  secret_id = var.wandb_api_key_secret

  replication {
    auto {}
  }

  labels = merge(var.labels, {
    component   = "wandb"
    environment = var.environment
  })
}

resource "google_secret_manager_secret" "github_token" {
  secret_id = var.github_token_secret

  replication {
    auto {}
  }

  labels = merge(var.labels, {
    component   = "github"
    environment = var.environment
  })
}

# Secret versions (placeholders - actual values should be set via Secret Manager)
# For dev/staging: placeholder values are created so terraform apply succeeds.
# Replace with real secrets before using: gcloud secrets versions add <secret-id> --data-file=-
# For prod: empty string is used; real values must be added manually via Secret Manager
resource "google_secret_manager_secret_version" "modal_token" {
  secret      = google_secret_manager_secret.modal_token.id
  secret_data = var.environment == "prod" ? "" : "dev-modal-token-placeholder"
}

resource "google_secret_manager_secret_version" "wandb_api_key" {
  secret      = google_secret_manager_secret.wandb_api_key.id
  secret_data = var.environment == "prod" ? "" : "dev-wandb-key-placeholder"
}

resource "google_secret_manager_secret_version" "github_token" {
  secret      = google_secret_manager_secret.github_token.id
  secret_data = var.environment == "prod" ? "" : "dev-github-token-placeholder"
}

# IAM Roles for Modal Runner
resource "google_project_iam_member" "modal_runner_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.modal_runner.email}"
}

resource "google_project_iam_member" "modal_runner_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.modal_runner.email}"
}

resource "google_project_iam_member" "modal_runner_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.modal_runner.email}"
}

resource "google_project_iam_member" "modal_runner_monitoring_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.modal_runner.email}"
}

# IAM Roles for GitHub Actions Service Account
resource "google_project_iam_member" "github_actions_storage_admin" {
  count   = var.enable_workload_identity ? 1 : 0
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.github_actions[0].email}"
}

resource "google_project_iam_member" "github_actions_secret_accessor" {
  count   = var.enable_workload_identity ? 1 : 0
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.github_actions[0].email}"
}

resource "google_project_iam_member" "github_actions_artifact_registry_writer" {
  count   = var.enable_workload_identity ? 1 : 0
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.github_actions[0].email}"
}

resource "google_project_iam_member" "github_actions_cloud_build_editor" {
  count   = var.enable_workload_identity ? 1 : 0
  project = var.project_id
  role    = "roles/cloudbuild.builds.editor"
  member  = "serviceAccount:${google_service_account.github_actions[0].email}"
}

resource "google_project_iam_member" "github_actions_service_account_user" {
  count   = var.enable_workload_identity ? 1 : 0
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.github_actions[0].email}"
}

# IAM Roles for Cloud Build Service Account
resource "google_project_iam_member" "cloud_build_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}

resource "google_project_iam_member" "cloud_build_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}

resource "google_project_iam_member" "cloud_build_artifact_registry_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}

resource "google_project_iam_member" "cloud_build_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}

resource "google_project_iam_member" "cloud_build_service_account_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}

# Grant Storage Admin to specified users
resource "google_project_iam_member" "storage_admin_users" {
  for_each = toset(var.storage_admin_emails)
  project  = var.project_id
  role     = "roles/storage.admin"
  member   = "user:${each.value}"
}

# Artifact Registry Repository for Docker images
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "swe-qwen-${var.environment}"
  description   = "Docker repository for SWE-Qwen images (${var.environment})"
  format        = "DOCKER"

  labels = merge(var.labels, {
    environment = var.environment
  })
}

# Grant Artifact Registry access to service accounts
resource "google_artifact_registry_repository_iam_member" "modal_runner_writer" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.docker_repo.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.modal_runner.email}"
}

resource "google_artifact_registry_repository_iam_member" "github_actions_writer" {
  count      = var.enable_workload_identity ? 1 : 0
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.docker_repo.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.github_actions[0].email}"
}

resource "google_artifact_registry_repository_iam_member" "cloud_build_writer" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.docker_repo.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.cloud_build.email}"
}

# Bucket IAM bindings for Modal Runner
resource "google_storage_bucket_iam_member" "dataset_writer" {
  count  = var.dataset_bucket_name != "" && var.environment != "prod" ? 1 : 0
  bucket = var.dataset_bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.modal_runner.email}"
}

resource "google_storage_bucket_iam_member" "model_writer" {
  count  = var.model_bucket_name != "" && var.environment != "prod" ? 1 : 0
  bucket = var.model_bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.modal_runner.email}"
}

# Public read access for datasets (dev only)
resource "google_storage_bucket_iam_member" "dataset_public_read" {
  count  = var.dataset_bucket_name != "" && var.environment == "dev" ? 1 : 0
  bucket = var.dataset_bucket_name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}
