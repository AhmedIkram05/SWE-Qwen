# GCS Storage Buckets for SWE-Qwen Platform
# Creates dataset and model checkpoint buckets with proper IAM and lifecycle

# Dataset bucket - auto-generate name if not provided
resource "random_string" "dataset_suffix" {
  length  = 6
  special = false
  upper   = false
  keepers = {
    project_id  = var.project_id
    environment = var.environment
  }
}

resource "google_storage_bucket" "dataset" {
  name                        = var.dataset_bucket != "" ? var.dataset_bucket : "${var.project_id}-${var.environment}-datasets-${random_string.dataset_suffix.result}"
  project                     = var.project_id
  location                    = var.region
  force_destroy               = var.environment != "prod"
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age                = 365
      num_newer_versions = 3
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age                   = 90
      matches_storage_class = ["STANDARD"]
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
    condition {
      age                   = 365
      matches_storage_class = ["NEARLINE", "STANDARD"]
    }
  }

  labels = merge(var.labels, {
    purpose     = "datasets"
    environment = var.environment
  })
}

# Model checkpoint bucket - auto-generate name if not provided
resource "random_string" "model_suffix" {
  length  = 6
  special = false
  upper   = false
  keepers = {
    project_id  = var.project_id
    environment = var.environment
  }
}

resource "google_storage_bucket" "model" {
  name                        = var.model_bucket != "" ? var.model_bucket : "${var.project_id}-${var.environment}-models-${random_string.model_suffix.result}"
  project                     = var.project_id
  location                    = var.region
  force_destroy               = var.environment != "prod"
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age                = 730
      num_newer_versions = 5
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age                   = 30
      matches_storage_class = ["STANDARD"]
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
    condition {
      age                   = 180
      matches_storage_class = ["NEARLINE", "STANDARD"]
    }
  }

  labels = merge(var.labels, {
    purpose     = "model-checkpoints"
    environment = var.environment
  })
}
