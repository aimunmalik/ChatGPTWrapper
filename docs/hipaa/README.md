# HIPAA compliance documentation — ANNA Health

This folder holds ANNA Health's HIPAA administrative paperwork for **Praxis**, an internal clinical AI assistant built on the `anna-chat` codebase. Praxis runs on AWS (Bedrock, Lambda, DynamoDB, Cognito, CloudFront) and may process Protected Health Information (PHI).

These documents are the *administrative safeguards* side of the Security Rule. The *technical* safeguards are implemented in code and described in [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) and [`docs/HIPAA_NOTES.md`](../HIPAA_NOTES.md).

> **These are starting templates.** ANNA's compliance lead and external counsel must review before final signature. Nothing in this folder constitutes a guarantee of HIPAA compliance.

## Document index

| # | Document | Regulation | Status |
|---|---|---|---|
| 1 | [RISK_ASSESSMENT.md](RISK_ASSESSMENT.md) | §164.308(a)(1)(ii)(A) | Draft |
| 2 | [POLICIES/ACCESS_CONTROL.md](POLICIES/ACCESS_CONTROL.md) | §164.308(a)(4), §164.312(a)(1) | Draft |
| 3 | [POLICIES/INCIDENT_RESPONSE.md](POLICIES/INCIDENT_RESPONSE.md) | §164.308(a)(6), §164.404 | Draft |
| 4 | [POLICIES/WORKFORCE_TRAINING.md](POLICIES/WORKFORCE_TRAINING.md) | §164.308(a)(5) | Draft |
| 5 | [POLICIES/SANCTIONS.md](POLICIES/SANCTIONS.md) | §164.308(a)(1)(ii)(C) | Draft |
| 6 | [POLICIES/CONTINGENCY.md](POLICIES/CONTINGENCY.md) | §164.308(a)(7) | Draft |
| 7 | [BAA_TRACKER.md](BAA_TRACKER.md) | §164.308(b)(1), §164.502(e) | Draft |

Statuses: **Draft** → **Approved** (by ANNA compliance lead) → **Signed** (wet or e-sig on file).

## Where signed copies live

These markdown files are the *working drafts*. Once approved and signed, the countersigned PDFs are filed in **ANNA's compliance record system** (`ANNA Google Drive → Compliance → HIPAA (restricted folder)`) — not in Git. Git holds the templates and their version history; the compliance record system holds the legally operative documents.

## Review cadence

- Annual review of every document by the HIPAA Security Officer and Privacy Officer.
- Ad-hoc review on any material change: new AWS service added, new vendor, new user role, after any reportable incident, or after a regulatory update.
- Track review dates in each document's signature block.

## Officers of record

| Role | Name | Effective |
|---|---|---|
| HIPAA Security Officer | `Aimun Malik`, `Founder / HIPAA Security & Privacy Officer` | `[DATE]` |
| HIPAA Privacy Officer | `Aimun Malik`, `Founder / HIPAA Security & Privacy Officer` | `[DATE]` |

Both officers must be named on record with ANNA HR before any Praxis production deploy.

## Reminder

- Templates are skimmable-but-real. Do not treat them as legal advice.
- Fill every `[PLACEHOLDER]` before counter-signing.
- External counsel should confirm state-level breach notification language (Illinois PIPA, California CCPA breach provisions, etc.) applies to ANNA's operating states before Section X of INCIDENT_RESPONSE.md is finalized.
