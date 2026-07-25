# Storage Module Outputs

output "dataset_bucket_name" {
  description = "Name of the dataset GCS bucket"
  value       = google_storage_bucket.dataset.name
}

output "dataset_bucket_self_link" {
  description = "Self link of the dataset GCS bucket"
  value       = google_storage_bucket.dataset.self_link
}

output "model_bucket_name" {
  description = "Name of the model checkpoint GCS bucket"
  value       = google_storage_bucket.model.name
}

output "model_bucket_self_link" {
  description = "Self link of the model checkpoint GCS bucket"
  value       = google_storage_bucket.model.self_link
}

output "dataset_bucket_location" {
  description = "Location of the dataset bucket"
  value       = google_storage_bucket.dataset.location
}

output "model_bucket_location" {
  description = "Location of the model checkpoint bucket"
  value       = google_storage_bucket.model.location
}
