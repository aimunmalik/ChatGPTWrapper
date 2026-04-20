# ──────────────────────────────────────────────────────────────────────────
# Custom domain for Praxis — praxis.annaautismcare.com
#
# Two-step rollout because annaautismcare.com DNS lives in Squarespace, not
# Route 53:
#
#   Step 1 (this commit): request the ACM cert and output the DNS validation
#   records. CloudFront is NOT yet attached to the custom domain.
#
#   Step 2 (next commit, after the cert validates): attach the custom domain
#   to the CloudFront distribution, update Cognito callback URLs, update the
#   API Gateway CORS list.
#
# Manual action between the two commits:
#   - Add the validation CNAME from output `acm_validation_records` to
#     Squarespace DNS.
#   - Wait for ACM to report `ISSUED` (usually 2-30 min).
#   - Add the `praxis` CNAME pointing at the CloudFront domain.
# ──────────────────────────────────────────────────────────────────────────

locals {
  app_fqdn = "praxis.annaautismcare.com"
}

resource "aws_acm_certificate" "app" {
  domain_name       = local.app_fqdn
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = merge(local.tags, {
    Name = local.app_fqdn
  })
}

output "app_fqdn" {
  description = "The custom domain we're provisioning. Add a CNAME at this name pointing to the CloudFront domain (see `cloudfront_url` output) once the ACM cert is issued."
  value       = local.app_fqdn
}

output "acm_validation_records" {
  description = "DNS validation records to add in Squarespace. Each entry is {name, type, value}. Add the CNAME Squarespace wants."
  value = [
    for dvo in aws_acm_certificate.app.domain_validation_options : {
      name  = dvo.resource_record_name
      type  = dvo.resource_record_type
      value = dvo.resource_record_value
    }
  ]
}
