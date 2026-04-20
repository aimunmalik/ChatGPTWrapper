variable "alias_suffix" {
  description = "Suffix appended to 'alias/anna-chat-' for the KMS alias. Example: 'dev-dynamodb' -> 'alias/anna-chat-dev-dynamodb'."
  type        = string
}

variable "description" {
  description = "Human-readable description of what this key protects."
  type        = string
}

variable "deletion_window_in_days" {
  description = "Waiting period before the key is actually deleted after scheduling deletion."
  type        = number
  default     = 30
}

variable "service_principals" {
  description = "AWS service principals allowed to use this key (e.g. logs.us-east-1.amazonaws.com)."
  type        = list(string)
  default     = []
}

variable "allowed_iam_principal_arns" {
  description = "IAM principal ARNs (roles/users) allowed to use this key."
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags to apply to the key."
  type        = map(string)
  default     = {}
}
