# Workforce Access Control Policy — Praxis

**Regulation:** 45 CFR §164.308(a)(4) (Information Access Management), §164.312(a)(1) (Access Control)
**Applies to:** Praxis (anna-chat) and all AWS resources supporting it
**Document owner:** HIPAA Security Officer
**Effective date:** `[DATE]`

---

## 1. Purpose

Defines who may access Praxis, how access is granted, reviewed, and revoked. The goal is **least privilege** — every workforce member gets only the access they need to do their clinical or operational work, nothing more.

## 2. Who can access Praxis

Access is limited to ANNA Health workforce members (employees, contractors) who:

1. Have completed HIPAA workforce training ([WORKFORCE_TRAINING.md](WORKFORCE_TRAINING.md)).
2. Have signed the ANNA confidentiality agreement.
3. Appear on the **Praxis Access Roster** maintained by the HIPAA Security Officer.

The roster is reviewed **quarterly** (see Section 7). No one outside the roster is granted a Cognito user, regardless of tenure or seniority.

## 3. Authentication

**Unique user IDs.** Every workforce member authenticates through Amazon Cognito with their own identity — no shared accounts, no service accounts used by humans. Cognito `sub` is the canonical user ID in every log line and every DynamoDB row.

**Multi-factor authentication.** TOTP MFA is **required** on every Cognito user at first login and on every subsequent sign-in. SMS-based MFA is not accepted (vulnerable to SIM swap).

**Session lifetime.** Access tokens are valid for **1 hour**. Refresh tokens are valid for the Cognito default (30 days) but revocable by an admin. Inactive sessions must be re-authenticated.

**Password requirements.** Cognito policy enforces: minimum 12 characters, upper + lower + number + symbol, no breached-password reuse (Cognito risk-based auth). Password rotation not forced on a schedule (NIST 800-63B), but forced on any suspected compromise.

## 4. Role-based groups

Two Cognito groups today:

| Group | What it can do |
|---|---|
| `users` | Sign in, create conversations, send messages, view own conversation history. |
| `admins` | Everything `users` can do, plus view the access roster, invite new users, disable users, read audit summaries. |

Neither group grants raw AWS console access. AWS console access is a separate control plane governed by IAM Identity Center (see Section 5).

Membership in `admins` is limited to the HIPAA Security Officer and explicitly-named backups approved in writing.

## 5. Infrastructure access (AWS console / CLI)

Praxis is built on AWS. Engineers who administer the infrastructure authenticate through **IAM Identity Center** (SSO), not through long-lived IAM user keys. Requirements:

- TOTP MFA required.
- No long-lived access keys for humans, ever.
- Time-bounded session credentials only.
- Least-privilege permission sets mapped to job function.

Application users (clinicians) have **no** AWS console or CLI access.

## 6. Provisioning

1. Manager submits a Praxis access request to the HIPAA Security Officer, including job function and business need.
2. Security Officer confirms training complete + NDA on file.
3. Admin invites the user in Cognito — user receives a one-time invite link, sets a password, and enrolls TOTP at first login.
4. Admin adds the user to the appropriate group (`users` or `admins`).
5. Security Officer updates the access roster.

## 7. Periodic review

The HIPAA Security Officer runs a **quarterly access audit**:

- Pull the full Cognito user list.
- Compare to the HR roster of active workforce members.
- Flag any Cognito user who is no longer on the HR roster — disable immediately.
- Flag any user in `admins` — confirm the role is still justified.
- Confirm MFA is enrolled on every active user.
- Record the review (date, reviewer, findings) in the compliance record system.

Review findings that show a lapse are a reportable event under [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) Section 2.

## 8. Termination / separation

When a workforce member separates (voluntary or involuntary):

- HR notifies the HIPAA Security Officer (or backup) at or before the separation time.
- **Within 4 hours of separation**, the admin disables the user in Cognito (`admin-disable-user`). Disabled — not deleted — so audit records remain intact.
- Within 24 hours, the admin removes the user from the access roster and any IAM Identity Center permission sets.
- Within 72 hours, Security Officer confirms no orphaned API keys or outstanding sessions by checking CloudTrail for activity after the disable timestamp.

Emergency separation (e.g. terminated for cause, credential compromise suspected): disable **immediately** upon notification — no 4-hour window.

## 9. Emergency access

If a Praxis admin is unavailable and access must be granted or revoked urgently, the HIPAA Security Officer (or designated backup) has standing authority to act and must document the action within 24 hours.

## 10. Related documents

- [Workforce training](WORKFORCE_TRAINING.md)
- [Sanctions policy](SANCTIONS.md)
- [Incident response](INCIDENT_RESPONSE.md)
- [Technical HIPAA notes](../../HIPAA_NOTES.md)

## 11. Signatures

| Role | Name | Signature | Date |
|---|---|---|---|
| HIPAA Security Officer | `Aimun Malik`, `Founder / HIPAA Security & Privacy Officer` | ____________________ | `[DATE]` |
| HIPAA Privacy Officer | `Aimun Malik`, `Founder / HIPAA Security & Privacy Officer` | ____________________ | `[DATE]` |

Effective date: `[DATE]`. Next review: `[DATE + 1 year]`.
