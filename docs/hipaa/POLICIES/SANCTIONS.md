# Sanctions Policy — Praxis and PHI handling

**Regulation:** 45 CFR §164.308(a)(1)(ii)(C) (Sanction Policy)
**Applies to:** Every ANNA Health workforce member with access to Praxis or any other PHI
**Document owner:** HIPAA Privacy Officer (in coordination with HR)
**Effective date:** `[DATE]`

---

## 1. Purpose

HIPAA requires ANNA Health to "apply appropriate sanctions against workforce members who fail to comply with the security policies and procedures of the covered entity." This policy describes those sanctions and gives concrete examples so workforce members know where the lines are.

Sanctions are not about punishment for the sake of punishment — they are about protecting patients and maintaining the trust that lets ANNA do its clinical work. A fair, predictable sanctions policy is part of how we keep PHI safe.

## 2. Scope

Applies to:

- Full-time and part-time ANNA employees
- Contractors and temporary workers with Praxis access
- Student interns with Praxis access
- Any other workforce member as defined under §160.103

Does **not** apply to patients, family members, or external parties — those are governed by separate civil and criminal law.

## 3. Progressive discipline

Sanctions escalate with the severity of the violation and with prior offenses. The Privacy Officer, with HR, determines the appropriate level for each incident. Mitigating and aggravating factors — intent, prior record, cooperation, actual harm — are weighed on the record.

| Level | Typical response |
|---|---|
| **1 — Verbal warning** | Documented conversation with the employee and their manager. Retraining assigned. |
| **2 — Written warning** | Formal letter, filed in personnel record. Retraining required. May include temporary access suspension. |
| **3 — Suspension** | Paid or unpaid suspension. Praxis access disabled during the suspension window. |
| **4 — Termination** | Separation per [ACCESS_CONTROL.md](ACCESS_CONTROL.md) §8 emergency termination. Referral to external counsel for civil / criminal assessment. Notification to HHS or state licensing boards where required. |

A first offense may jump directly to Level 4 if the violation is egregious (see Section 5 — *Termination-level*).

## 4. Mandatory reporting of violations

Every workforce member has a **duty to report** suspected violations of this policy — their own or a colleague's — to the HIPAA Privacy Officer or Security Officer without delay. Knowingly failing to report a known violation is itself a violation subject to sanction at Level 2 or higher.

Retaliation against a workforce member for reporting in good faith is prohibited and itself a terminable offense.

## 5. Concrete examples

These are illustrative — not exhaustive. Real-world situations are judged on their facts.

### 5.1 Minor — Level 1 (verbal warning)

- Using Praxis briefly for a personal task (asking for a vacation itinerary, a recipe, etc.). Not ideal, not a privacy risk, but a misuse of a clinical system. First offense → verbal warning and a reminder that Praxis is for clinical use only.
- Leaving a laptop unlocked at an ANNA workstation for a short period with no actual exposure. Verbal warning + reminder of session-hygiene training.
- One-off structured-logging lapse in a dev environment caught in code review before merge.

### 5.2 Moderate — Level 2 (written warning)

- Repeated minor violations after a verbal warning.
- Discussing a specific Praxis conversation with a colleague who has no clinical need.
- Emailing Praxis content to a personal email "to read later" (creates an unsecured copy outside the BAA boundary).
- Falling more than 30 days behind on annual training refresh.

### 5.3 Major — Level 3 (suspension)

- **Sharing credentials.** Giving your Cognito password, TOTP seed, or active session token to anyone, for any reason. This is a hard line. Shared credentials destroy audit integrity and are treated as a major offense regardless of intent.
- Bypassing MFA (attempts to share TOTP codes, register another person's device).
- Using Praxis to search for information on a person who is not a current clinical contact (family member, friend, public figure, ex-partner).
- Posting Praxis screenshots in non-approved channels (personal Slack, group chats, social media) even without obvious PHI.

### 5.4 Termination-level — Level 4 (termination + potential legal)

- **Knowingly exposing PHI.** Deliberately disclosing patient information to an unauthorized party, whether for personal gain, curiosity, or malice.
- Selling PHI or trading it for consideration of any kind.
- Falsifying audit records or tampering with CloudTrail, logs, or access rosters.
- Intentionally bypassing technical safeguards (e.g. exporting DynamoDB data through a back channel).
- Repeated major offenses after prior suspension.

## 6. Due process

Before a sanction at Level 2 or higher issues:

1. The Privacy Officer and HR document the facts.
2. The workforce member is informed of the allegation and given an opportunity to respond, in a private meeting with a union or support representative if applicable.
3. A written determination is issued, referencing the evidence and the applicable level.
4. The determination and the workforce member's response are filed in the compliance record system.

## 7. Documentation and retention

Sanctions records are retained for **6 years** from the date of action, per §164.316(b)(2). Access to sanctions records is limited to HR, the Privacy Officer, the Security Officer, and legal counsel.

## 8. Related documents

- [Workforce training](WORKFORCE_TRAINING.md)
- [Access control](ACCESS_CONTROL.md)
- [Incident response](INCIDENT_RESPONSE.md)

## 9. Signatures

| Role | Name | Signature | Date |
|---|---|---|---|
| HIPAA Privacy Officer | `[NAME]`, `[TITLE]` | ____________________ | `[DATE]` |
| HIPAA Security Officer | `[NAME]`, `[TITLE]` | ____________________ | `[DATE]` |
| HR representative | `[NAME]`, `[TITLE]` | ____________________ | `[DATE]` |

Effective date: `[DATE]`. Next review: `[DATE + 1 year]`.
