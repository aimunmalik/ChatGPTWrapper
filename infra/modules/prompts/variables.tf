variable "env" {
  description = "Environment name (dev, prod). Used as resource prefix."
  type        = string
}

variable "kms_key_arn" {
  description = "Customer-managed KMS key ARN used to encrypt the prompts DynamoDB table."
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
