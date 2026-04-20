variable "env" {
  description = "Environment name (dev, prod). Used as resource prefix."
  type        = string
}

variable "kms_key_arn" {
  description = "Customer-managed KMS key ARN used for SSE-KMS on the attachments bucket."
  type        = string
}

variable "dynamodb_kms_key_arn" {
  description = "Customer-managed KMS key ARN used to encrypt the attachments DynamoDB table."
  type        = string
}

variable "cors_allow_origins" {
  description = "Origins allowed by the attachments bucket CORS policy (CloudFront origin and local dev)."
  type        = list(string)
  default     = ["http://localhost:5173"]
}

variable "extract_lambda_arn" {
  description = "ARN of the extraction Lambda invoked by S3 ObjectCreated events under the attachments/ prefix."
  type        = string
}

variable "deletion_protection" {
  description = "Whether the DynamoDB table has deletion protection enabled. Should be true in prod."
  type        = bool
  default     = true
}

variable "guardduty_detector_id" {
  description = <<-EOT
    Optional ID of an existing account-level GuardDuty detector. GuardDuty
    detectors are a per-region singleton, so if the account already has one
    enabled (common when using AWS Organizations delegated admin), pass its
    ID here and this module will skip creating a new one. Leave empty to
    let the module create and manage a detector.
  EOT
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags applied to all resources created by this module."
  type        = map(string)
  default     = {}
}
