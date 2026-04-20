# Incident Response and Breach Notification Policy — Praxis

**Regulation:** 45 CFR §164.308(a)(6) (Security Incident Procedures), §164.404 (Breach Notification to Individuals), §164.408 (Notification to HHS), §164.410 (Notification by Business Associates)
**Applies to:** Praxis (anna-chat) and any PHI processed by it
**Document owner:** HIPAA Privacy Officer
**Effective date:** `[DATE]`

> The technical runbook for first-hour triage lives in [`docs/INCIDENT_RESPONSE.md`](../../INCIDENT_RESPONSE.md). This document is the administrative policy that wraps around it.

---

## 1. Purpose

Defines how ANNA Health detects, responds to, and (where required) notifies affected parties about security incidents and breaches involving Praxis. HIPAA draws a bright line between an *incident* and a *breach* — this policy makes that line, and the clock that starts when it's crossed, explicit.

## 2. Definitions

**Security incident.** Any attempted or successful unauthorized access, use, disclosure, modification, or destruction of information, or interference with system operations. A failed phishing attempt against a Praxis user is an incident. Most incidents are not breaches.

**Breach.** An unauthorized acquisition, access, use, or disclosure of unsecured PHI that compromises its security or privacy, under §164.402. *Secured* PHI (encrypted per HHS guidance, which Praxis PHI is at rest and in transit) is generally not a breach when only ciphertext is exposed.

**Discovery.** A breach is "discovered" the first moment *any* ANNA workforce member knows, or by exercising reasonable diligence would have known, of its occurrence. The 60-day clock starts at discovery — not at confirmation.

## 3. Roles

| Role | Responsibility |
|---|---|
| **HIPAA Security Officer** | Triages every incident. Runs the technical runbook. Preserves evidence. |
| **HIPAA Privacy Officer** | Makes the call on whether an incident is a breach. Decides notification. Handles HHS filings. |
| **Engineering on-call** | Executes containment steps at the direction of the Security Officer. |
| **External counsel** | Reviews breach notification wording and state-law obligations before notice issues. |
| **ANNA leadership** | Notified on any confirmed breach affecting ≥ 1 individual. |

## 4. Response lifecycle

### 4.1 Detection

Incidents surface from any of: GuardDuty findings, CloudWatch alarms, user reports, workforce reports, or external notification (e.g. AWS abuse, security researcher). Every workforce member is required to report suspected incidents immediately per Section 6.

### 4.2 Triage

The Security Officer, within **1 hour of report**:

- Opens an incident ticket with a precise `discovery_timestamp`.
- Classifies severity: SEV-1 (confirmed PHI exposure), SEV-2 (suspected), SEV-3 (technical only, no PHI risk).
- Notifies the Privacy Officer on SEV-1 or SEV-2.
- Follows [`docs/INCIDENT_RESPONSE.md`](../../INCIDENT_RESPONSE.md) Section "The first hour."

### 4.3 Containment

Stop ongoing harm. Typical moves: disable a Cognito user, rotate an IAM role, set Lambda reserved concurrency to zero, disable a KMS key. Preserve evidence — do not delete logs, Lambda versions, or backups.

### 4.4 Eradication

Remove the root cause. Patch vulnerable code paths. Rotate exposed credentials. Remove attacker persistence (Cognito users, IAM roles, scheduled events).

### 4.5 Recovery

Restore service from clean state. If data integrity was compromised, restore from DynamoDB PITR. Run a fresh security check before re-enabling user access.

### 4.6 Lessons learned

Within **7 days** of incident close, the Security Officer writes an incident report and runs a blameless post-mortem. Detection gaps update GuardDuty / AWS Config rules. Workforce-error contributions feed a training item under [WORKFORCE_TRAINING.md](WORKFORCE_TRAINING.md).

## 5. Breach notification

If the Privacy Officer, after risk assessment under §164.402(2), determines an incident is a breach:

### 5.1 Notification to affected individuals — §164.404

- **Deadline:** without unreasonable delay, and **no later than 60 calendar days from discovery**.
- **Method:** written notice by first-class mail (or e-mail if the individual has agreed to electronic notice).
- **Content:** brief description of what happened, date of breach, date of discovery, types of PHI involved, steps individuals should take to protect themselves, what ANNA is doing, contact information.

### 5.2 Notification to HHS — §164.408

- **≥ 500 individuals:** notify HHS Secretary concurrently with individual notice (within 60 days).
- **< 500 individuals:** log in the annual HHS breach report, submitted **within 60 days of the end of the calendar year** in which the breach was discovered.

### 5.3 Notification to media — §164.406

- **≥ 500 individuals in a single state or jurisdiction:** notify prominent media outlets serving that area, without unreasonable delay and no later than 60 days from discovery.

### 5.4 State law

In addition to HIPAA, ANNA is subject to state breach notification laws in every state where affected individuals reside. These may impose shorter deadlines or additional content requirements than HIPAA.

> **Placeholder — external counsel to populate before final signature.** Confirm applicable state laws for ANNA's operating footprint (e.g. `[STATE]` breach notification statute) and document overrides here.

## 6. Workforce reporting obligation

Every workforce member must report any suspected security incident **immediately** — at minimum, within 1 hour of noticing it. Report by `[ANNA CONTACT METHOD]` (e.g. a dedicated Slack channel, a ticketing address, a phone tree). Failure to report is a sanction-eligible violation under [SANCTIONS.md](SANCTIONS.md).

Non-retaliation: no workforce member will face retaliation for reporting in good faith.

## 7. Contacts

Populate at activation; re-verify quarterly.

| Role | Name | Phone | Email |
|---|---|---|---|
| HIPAA Security Officer | `[NAME]` | `[PHONE]` | `[EMAIL]` |
| HIPAA Privacy Officer | `[NAME]` | `[PHONE]` | `[EMAIL]` |
| Engineering on-call | `[ANNA CONTACT]` | `[PHONE]` | `[EMAIL]` |
| External counsel | `[NAME / FIRM]` | `[PHONE]` | `[EMAIL]` |
| AWS Enterprise Support | `[CASE LINK / PHONE]` | | |
| Cyber insurance broker | `[NAME]` | `[PHONE]` | `[EMAIL]` |

## 8. Documentation

Every incident, whether or not it becomes a breach, is documented and retained for **6 years** from the later of: creation date or last effective date, per §164.316(b)(2).

## 9. Signatures

| Role | Name | Signature | Date |
|---|---|---|---|
| HIPAA Security Officer | `[NAME]`, `[TITLE]` | ____________________ | `[DATE]` |
| HIPAA Privacy Officer | `[NAME]`, `[TITLE]` | ____________________ | `[DATE]` |

Effective date: `[DATE]`. Next review: `[DATE + 1 year]`.
