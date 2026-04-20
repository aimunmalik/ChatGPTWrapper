output "vpc_id" {
  description = "ID of the VPC."
  value       = aws_vpc.this.id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC."
  value       = aws_vpc.this.cidr_block
}

output "private_subnet_ids" {
  description = "IDs of the private subnets. Attach Lambda here."
  value       = aws_subnet.private[*].id
}

output "availability_zones" {
  description = "AZs used."
  value       = [for s in aws_subnet.private : s.availability_zone]
}

output "vpc_endpoints_security_group_id" {
  description = "Security group ID for VPC endpoints. Lambda SGs must be allowed to reach this on 443."
  value       = aws_security_group.vpc_endpoints.id
}

output "route_table_id" {
  description = "ID of the private route table."
  value       = aws_route_table.private.id
}
