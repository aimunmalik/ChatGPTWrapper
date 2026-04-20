variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "aws_region" {
  description = "AWS region for VPC endpoint service names."
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC. /20 gives 4096 addresses, enough for per-AZ subnets with room to grow."
  type        = string
  default     = "10.20.0.0/20"
}

variable "flow_logs_enabled" {
  description = "Whether to enable VPC flow logs. Required posture for HIPAA environments; can be disabled in dev to save cost."
  type        = bool
  default     = true
}

variable "logs_kms_key_arn" {
  description = "KMS CMK ARN used to encrypt flow log groups."
  type        = string
  default     = null
}

variable "log_retention_days" {
  description = "Flow log retention in days."
  type        = number
  default     = 90
}

variable "tags" {
  description = "Tags applied to all resources in the module."
  type        = map(string)
  default     = {}
}
