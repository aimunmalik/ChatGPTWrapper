# Ops runbook

> **Status:** skeleton. Filled as each phase lands.

## On-call basics

- Primary signals: CloudWatch alarms, GuardDuty findings, Cognito anomalous sign-in alerts.
- No pager rotation yet — single on-call is the ANNA HIPAA Security Officer.
- Any `SEV-1` or `SEV-2` event → escalate to the HIPAA Privacy Officer within 1 hour.

## Severity levels

| Severity | Definition | Example |
|---|---|---|
| SEV-1 | Confirmed or suspected breach of PHI | Unauthorized API call observed in CloudTrail; bulk `Scan` on Messages table by unknown principal |
| SEV-2 | Service is down OR strong indicator of imminent breach | Chat streaming returning 5xx for >5 min; GuardDuty `UnauthorizedAccess` finding |
| SEV-3 | Degraded service, no PHI exposure risk | Elevated latency; rate-limit tripping for legitimate user |
| SEV-4 | Cosmetic or internal | Log noise, drift in non-sensitive AWS Config rule |

## Quarterly access review (required)

Performed by: HIPAA Security Officer.
Cadence: first week of Jan / Apr / Jul / Oct.
Checklist:

- [ ] Cognito user list matches ANNA current staff list
- [ ] Cognito users not logged in for 90+ days → disable
- [ ] IAM Identity Center users reviewed, unused roles removed
- [ ] IAM roles reviewed for least privilege drift
- [ ] KMS key policies unchanged from IaC baseline
- [ ] CloudTrail enabled and logs arriving
- [ ] GuardDuty enabled, no unacknowledged findings
- [ ] Document completion date in ANNA compliance records

## Routine checks

*Filled as phases land. Intended to include:*

- Daily: check CloudWatch alarm state, GuardDuty findings
- Weekly: review Bedrock token usage, DynamoDB capacity metrics
- Monthly: review AWS Config rule compliance, patch Lambda runtimes

## Monitoring and alerts

*Filled in Phase 5. Planned alerts:*

- Chat streaming Lambda error rate > 2%
- P95 latency > 10s
- Bedrock throttling events
- Cognito sign-in failures > threshold from single IP
- GuardDuty high-severity findings → immediate page
- AWS Config non-compliance with HIPAA conformance pack

## Future improvements tracked

- Split dev and prod into separate AWS Organizations accounts; apply SCPs to prevent cross-environment access
- Multi-region DR (DynamoDB global tables, Lambda replicas in us-west-2)
- Custom domain with ACM certificate
- Automated pen test in CI (weekly OWASP ZAP scan against dev)
