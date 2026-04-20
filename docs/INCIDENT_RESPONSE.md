# Incident response — breach of PHI checklist

> This runbook is the technical companion to ANNA's written HIPAA Incident Response Plan. It does not replace that plan. If the two disagree, the written plan controls.

## When to use this

Use this checklist the moment any of the following is observed or suspected:

- GuardDuty finding of severity `HIGH` on any anna-chat resource
- CloudTrail shows API calls from an unexpected principal or region
- DynamoDB `Scan` or large `Query` by anything other than the known Lambda role
- A user reports seeing another user's conversation
- Lost/stolen device with cached Cognito tokens for anna-chat
- Any indicator that a KMS key was used outside its policy scope

## The first hour (technical triage)

1. **Preserve evidence.** Do NOT delete logs, Lambda versions, CloudTrail files, or DynamoDB backups. Every artifact is potential evidence for the incident report.
2. **Identify the scope.**
   - Which resources are affected?
   - Is the actor still active? Check CloudTrail for the principal's last event.
   - What PHI was touched? Use DynamoDB item counts + CloudWatch metrics, not content queries.
3. **Contain.**
   - If the attacker is using a compromised Cognito user → disable the user in the Cognito console.
   - If the attacker is using a compromised IAM role → rotate the role's trust policy.
   - If a Lambda is compromised → set its reserved concurrency to 0.
   - If a KMS key is suspect → disable it (not delete).
4. **Notify.**
   - HIPAA Privacy Officer: within 1 hour of discovery.
   - HIPAA Security Officer: within 1 hour of discovery.
   - Anthropic / AWS: via support if their services are implicated.

## Breach notification clock

Under HIPAA §164.404, breaches affecting ≥500 individuals require notification to:
- Affected individuals: within 60 days of discovery.
- HHS Secretary: concurrent with individual notification.
- Prominent media outlets (for breaches in a state/jurisdiction affecting ≥500): within 60 days.

Under §164.410, business associates must notify the covered entity within 60 days.

**Track `discovery_timestamp` precisely** — the countdown starts when any ANNA workforce member first knew, or reasonably should have known.

## Evidence collection

Collect and preserve for the incident report:

- CloudTrail log files covering the time window (download from the Object-Locked S3 bucket)
- CloudWatch Log streams for the Lambdas involved
- GuardDuty findings JSON
- Cognito user activity (sign-in events)
- DynamoDB item-level CloudWatch metrics
- List of KMS `Decrypt` operations from CloudTrail data events
- Current Terraform state (as of incident) — `terraform state pull > incident-<date>-state.json`

Store in a dedicated S3 bucket with Object Lock, CMK-encrypted, access restricted to the Security Officer.

## Recovery

1. Remove the attacker's access (completed in containment).
2. Rotate credentials that may have been exposed:
   - KMS key → rotate (automatic annual rotation does this)
   - Cognito user passwords → force reset
   - IAM roles → update trust policies
3. Restore DynamoDB from PITR if data integrity was compromised.
4. Deploy patches to any vulnerable code paths.
5. Run a fresh pen test pass before re-enabling user access.

## Post-incident

- Write the incident report within 7 days.
- Conduct blameless post-mortem; update this runbook with new detection signals if any were missed.
- Update AWS Config rules or GuardDuty suppression list if the incident exposed a gap.
- File a training item if workforce error contributed.

## Contacts

*To be filled by ANNA:*

- HIPAA Privacy Officer: `[name, phone, email]`
- HIPAA Security Officer: `[name, phone, email]`
- AWS Enterprise Support (if on Business/Enterprise plan): `[case link]`
- External counsel for breach notification: `[name, phone]`
