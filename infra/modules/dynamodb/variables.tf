variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "kms_key_arn" {
  description = "Customer-managed KMS key ARN used for table encryption."
  type        = string
}

variable "deletion_protection" {
  description = "Whether table deletion protection is enabled. Should be true in prod."
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags applied to all resources in the module."
  type        = map(string)
  default     = {}
}
