output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID (use for invalidations)."
  value       = aws_cloudfront_distribution.spa.id
}

output "cloudfront_domain_name" {
  description = "CloudFront default domain (e.g. d1234.cloudfront.net)."
  value       = aws_cloudfront_distribution.spa.domain_name
}

output "cloudfront_hosted_zone_id" {
  description = "Hosted zone ID (useful for Route 53 alias records later)."
  value       = aws_cloudfront_distribution.spa.hosted_zone_id
}

output "cloudfront_url" {
  description = "Full https:// URL of the CloudFront distribution."
  value       = "https://${aws_cloudfront_distribution.spa.domain_name}"
}

output "spa_bucket_name" {
  description = "Name of the SPA S3 bucket."
  value       = aws_s3_bucket.spa.id
}

output "spa_bucket_arn" {
  description = "ARN of the SPA S3 bucket."
  value       = aws_s3_bucket.spa.arn
}

output "waf_web_acl_arn" {
  description = "ARN of the WAF WebACL attached to CloudFront."
  value       = aws_wafv2_web_acl.spa.arn
}
