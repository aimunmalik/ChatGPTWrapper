# Contingency Plan — Praxis

**Regulation:** 45 CFR §164.308(a)(7) (Contingency Plan) — covering Data Backup Plan, Disaster Recovery Plan, Emergency Mode Operation Plan, Testing and Revision Procedures, and Applications and Data Criticality Analysis
**Applies to:** Praxis (anna-chat) and the AWS resources supporting it
**Document owner:** HIPAA Security Officer (in coordination with Engineering)
**Effective date:** `[DATE]`

---

## 1. Purpose

HIPAA requires a written contingency plan to respond to events that damage systems containing PHI. This plan covers backup, disaster recovery, emergency operation, and testing for Praxis.

## 2. Applications and data criticality analysis

Criticality ranking drives recovery priority. Higher criticality = shorter tolerated outage.

| Asset | Contains PHI | Criticality | RTO target | RPO target |
|---|---|---|---|---|
| Cognito User Pool | No (identity only) | Critical | 1 hour | 0 (AWS-managed) |
| API Gateway + Lambdas | Transient | Critical | 1 hour | 0 (stateless, IaC-redeployable) |
| DynamoDB — Messages | **Yes** | Critical | 4 hours | 5 minutes (PITR) |
| DynamoDB — Conversations | Yes (metadata) | Critical | 4 hours | 5 minutes (PITR) |
| Bedrock model access | No | Critical | 1 hour | 0 |
| KMS CMKs | No (key material) | Critical | Immediate | 0 (AWS-durable) |
| CloudTrail audit trail | No | High | 24 hours | 0 (continuous) |
| CloudWatch Logs | No | Medium | 24 hours | n/a |
| S3 attachments (Phase 6+) | Yes | Critical | 4 hours | 0 (versioned + Object Lock) |
| SPA static assets | No | High | 1 hour | 0 (redeployable from Git) |
| IaC (Terraform) in Git | No | Critical | Immediate | 0 (mirrored) |

**RTO** = Recovery Time Objective — how fast we get back online. **RPO** = Recovery Point Objective — how much data loss is acceptable.

## 3. Data backup plan

Required under §164.308(a)(7)(ii)(A). The goal is to be able to reconstruct every piece of PHI Praxis holds.

| Data | Backup mechanism | Retention | Notes |
|---|---|---|---|
| DynamoDB (both tables) | Point-in-time recovery (PITR) | **35 days** (AWS max) | Continuous; any second in the window is recoverable |
| CloudTrail events | Shipped to Object-Locked S3 bucket, CMK-encrypted | **7 years** | Immutable — cannot be deleted or modified |
| CloudWatch Logs | Native log group retention | 90 days | Metadata only; no PHI |
| IaC (Terraform) | Git; mirrored to GitHub | Forever | Full infrastructure definition |
| Lambda code | Versioned deployments + Git | Forever | Every deployed version retained |
| S3 attachments (Phase 6+) | S3 versioning + Object Lock | 90-day governance, 180-day lifecycle | See `docs/HIPAA_NOTES.md` — extensions require pre-change governance bump |
| KMS CMKs | AWS-managed durability | Annual rotation, 30-day deletion window | CMKs are held in ANNA's account; deletion requires explicit action |

Backups are themselves PHI-containing assets and encrypted accordingly. Access to restore from backup is limited to the Security Officer and a named engineering backup.

## 4. Disaster recovery plan

Required under §164.308(a)(7)(ii)(B).

### 4.1 Current posture

**Single-region deployment.** Praxis today runs in a single AWS region. A full-region outage means Praxis is unavailable until AWS restores the region. Historically AWS region-wide outages are measured in hours, not days.

This is acceptable for current clinical criticality (Praxis is an *assistant* — clinicians can do their job without it) but not optimal. Multi-region DR is tracked as a **Tier C roadmap item**.

### 4.2 Tier C — multi-region DR (future)

Target state, not current state:

- DynamoDB global tables across two regions.
- Route 53 health check + failover policy.
- Lambda deployed in both regions from the same IaC.
- Cognito user pool replicated via backup/restore (Cognito does not natively multi-region today).
- Cross-region CMK replicas.

Implementation triggered when: (a) Praxis becomes clinically essential (not just helpful), or (b) ANNA takes on a contract requiring a formal multi-region RTO.

### 4.3 Recovery procedure (current, single-region)

Scenarios and runbooks:

| Scenario | Procedure |
|---|---|
| Lambda code bug — bad deploy | Roll back to prior Lambda version; redeploy from last known-good Terraform state. ≤ 15 minutes. |
| DynamoDB data corruption (bad migration, rogue write) | Use PITR to restore to a point immediately before the corruption. ≤ 4 hours. |
| KMS key disabled in error | Re-enable; no data loss. ≤ 15 minutes. |
| AWS region outage — AWS service impaired | Wait for AWS recovery. Notify workforce per Section 5. |
| AWS region outage — prolonged (> 24 hours) | Escalate to Tier C multi-region spinup; requires ANNA leadership sign-off given time and cost. |
| Account compromise | Follow [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md); rotate credentials; restore from PITR if integrity compromised. |

## 5. Emergency mode operation plan

Required under §164.308(a)(7)(ii)(C). Describes how clinical operations continue when Praxis is unavailable.

### 5.1 Trigger

Praxis is considered in *emergency mode* when: it has been unavailable for **more than 4 hours** and recovery ETA is uncertain.

### 5.2 Actions

1. HIPAA Security Officer (or backup) notifies ANNA leadership and clinical leads within 30 minutes of crossing the 4-hour threshold.
2. Clinical leads notify workforce that Praxis is unavailable and to fall back to pre-Praxis workflows (direct review, consultation with supervisors, standard documentation in the EMR).
3. **No workforce member may substitute an unapproved AI tool** (consumer ChatGPT, Gemini, Copilot, consumer Claude) during the outage. Those tools are not covered by ANNA's BAA and using them for PHI is a Level 3+ violation under [SANCTIONS.md](SANCTIONS.md).
4. Security Officer maintains a running log: timestamp, status, actions taken, communications sent. Filed in the compliance record system after recovery.
5. On recovery, clinical leads confirm to workforce that Praxis is back. A post-incident review follows per [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) §4.6.

### 5.3 Protected essential operations

During emergency mode, the following continue **without** Praxis:

- Clinical documentation in the EMR.
- Patient scheduling and care coordination.
- Supervisor consultations and case review.
- All normal HIPAA-governed workflows ANNA used before Praxis existed.

Praxis is an accelerant. It is not on the critical path for patient safety.

## 6. Testing and revision

Required under §164.308(a)(7)(ii)(D).

- **Annual tabletop exercise.** The Security Officer leads a scripted walk-through of one disaster scenario (e.g. "DynamoDB corruption at 09:00 Monday"). Participants: Security Officer, Engineering on-call, Privacy Officer, clinical lead delegate.
- **Output:** written findings, including gaps and time-to-decision metrics. Filed in the compliance record system.
- **Quarterly backup verification.** Engineering runs a spot-check PITR restore to a throwaway DynamoDB table to confirm backups are usable. Synthetic data only in the dev account.
- **Plan update.** This document is revised on any material finding, on any actual incident that exercised the plan, and at minimum annually.

## 7. Related documents

- [Risk assessment](../RISK_ASSESSMENT.md)
- [Incident response](INCIDENT_RESPONSE.md)
- [Access control](ACCESS_CONTROL.md)
- [Technical HIPAA notes](../../HIPAA_NOTES.md)
- [Architecture](../../ARCHITECTURE.md)

## 8. Signatures

| Role | Name | Signature | Date |
|---|---|---|---|
| HIPAA Security Officer | `Aimun Malik`, `Founder / HIPAA Security & Privacy Officer` | ____________________ | `[DATE]` |
| HIPAA Privacy Officer | `Aimun Malik`, `Founder / HIPAA Security & Privacy Officer` | ____________________ | `[DATE]` |
| Engineering lead | `Aimun Malik`, `Founder / HIPAA Security & Privacy Officer` | ____________________ | `[DATE]` |

Effective date: `[DATE]`. Next review: `[DATE + 1 year]`.
