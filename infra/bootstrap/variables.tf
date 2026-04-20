variable "aws_region" {
  description = "AWS region for state storage. Must match the region the rest of the stack runs in."
  type        = string
  default     = "us-east-1"
}

variable "github_org" {
  description = "GitHub org or user that owns the repo (used in OIDC trust policy)."
  type        = string
  default     = "aimunmalik"
}

variable "github_repo" {
  description = "GitHub repo name (used in OIDC trust policy)."
  type        = string
  default     = "ChatGPTWrapper"
}
