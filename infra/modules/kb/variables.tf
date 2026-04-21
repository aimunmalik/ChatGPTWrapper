variable "env" {
  description = "Environment name (dev, prod). Used as resource prefix."
  type        = string
}

variable "kms_key_arn" {
  description = "Customer-managed KMS key ARN used for SSE-KMS on the knowledge-base bucket."
  type        = string
}

variable "dynamodb_kms_key_arn" {
  description = "Customer-managed KMS key ARN used to encrypt the knowledge-base DynamoDB table."
  type        = string
}

variable "cors_allow_origins" {
  description = "Origins allowed by the knowledge-base bucket CORS policy (CloudFront origin and local dev)."
  type        = list(string)
  default     = ["http://localhost:5173"]
}

variable "ingest_lambda_arn" {
  description = "ARN of the ingestion Lambda invoked by S3 ObjectCreated events under the kb/ prefix."
  type        = string
}

variable "deletion_protection" {
  description = "Whether the DynamoDB table has deletion protection enabled. Should be true in prod."
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags applied to all resources created by this module."
  type        = map(string)
  default     = {}
}
