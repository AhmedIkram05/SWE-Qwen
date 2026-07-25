variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "dataset_bucket" {
  description = "Name of the GCS bucket for datasets (empty = auto-generate)"
  type        = string
  default     = ""
}

variable "model_bucket" {
  description = "Name of the GCS bucket for model checkpoints (empty = auto-generate)"
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
