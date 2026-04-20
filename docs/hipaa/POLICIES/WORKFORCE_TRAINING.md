# Workforce Security Training Plan — Praxis

**Regulation:** 45 CFR §164.308(a)(5) (Security Awareness and Training)
**Applies to:** Every ANNA Health workforce member granted access to Praxis
**Document owner:** HIPAA Security Officer
**Effective date:** `[DATE]`

---

## 1. Purpose

HIPAA requires every workforce member who handles PHI to receive security awareness training appropriate to their role. This plan describes the curriculum, the delivery schedule, and the acknowledgment record for Praxis users.

## 2. When training happens

- **Before access is granted.** No Cognito user is provisioned for Praxis until training is complete and the acknowledgment form (Section 5) is signed.
- **Annually thereafter.** Every user refreshes training on or before the anniversary of their initial completion.
- **On material change.** Significant changes to Praxis (new features, new attachment handling, new vendor) trigger a targeted update session.

The HIPAA Security Officer tracks completion in the compliance record system. Access is revoked if a user's annual refresh is more than 30 days overdue.

## 3. Curriculum

Total delivery: ~45 minutes for initial, ~20 minutes for annual refresh. Format is flexible — recorded video, live session, or written self-study — as long as the acknowledgment is signed and a short quiz is passed.

### 3.1 HIPAA basics (10 min)

- What PHI is. Why ANNA is a covered entity.
- The Privacy Rule vs. the Security Rule at a high level.
- "Minimum necessary" — only use or disclose the PHI needed for the task.
- Patient rights (access, amend, accounting of disclosures) at a summary level.

### 3.2 Password hygiene and MFA (5 min)

- Password managers, unique passwords per service.
- Why TOTP MFA, not SMS.
- Do not share credentials. Sharing credentials is a major sanction-eligible offense ([SANCTIONS.md](SANCTIONS.md)).

### 3.3 Recognizing phishing and social engineering (10 min)

- Common phishing patterns in clinical settings.
- How to verify unexpected email asking for PHI, credentials, or a Cognito password reset.
- What to do if you click a suspicious link (disclose it — there is no penalty for honest reporting).

### 3.4 Reporting incidents (5 min)

- Every suspected security incident must be reported **immediately** per [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) Section 6.
- Report via `aimun@annaautismcare.com`.
- Non-retaliation guarantee for good-faith reports.

### 3.5 Praxis-specific guidance (15 min)

Praxis-specific rules are where AI-assistant clinical tools introduce new risk. These are not optional.

- **Do not paste PHI into prompts beyond what the conversation requires.** Praxis stores everything you type. Treat the text box like a clinical note — purposeful, minimum necessary, not a scratchpad.
- **Do not paste the same patient's data into any other AI tool** (ChatGPT, Gemini, Copilot, consumer Claude, any browser extension). Those services are not covered by ANNA's BAA. Praxis is the only approved AI destination for PHI.
- **No screenshots to unsecured channels.** Do not screenshot Praxis conversations and paste them into personal email, SMS, consumer chat apps, or social media.
- **No personal email.** Never forward Praxis content to a personal Gmail/Outlook/iCloud inbox. If you need something outside Praxis, use approved ANNA channels.
- **Session hygiene.** Lock your laptop when you step away. Sign out at end of day. 1-hour access tokens are an enforcement floor, not a reason to leave sessions open.
- **Attachments (when enabled in Phase 6+).** Only upload documents that are clinically relevant. Assume anything you upload is recorded. Do not upload documents you personally produced for non-ANNA reasons.
- **Report weirdness.** If Praxis responds with information you didn't ask for, or that looks like another patient's data, stop and report immediately per Section 3.4.

## 4. Assessment

A short quiz (≥ 80% to pass) confirms comprehension of each section. Failures re-test. Two failures in a row trigger one-on-one retraining with the HIPAA Security Officer.

## 5. Acknowledgment form

By signing below, I acknowledge that I have completed ANNA Health's HIPAA workforce security training for Praxis, understand the obligations described above (including the Praxis-specific rules), and understand that violations may result in sanctions up to and including termination and referral for legal action, as described in [SANCTIONS.md](SANCTIONS.md).

| Field | |
|---|---|
| Name | `Aimun Malik` |
| Title | `Founder / HIPAA Security & Privacy Officer` |
| Date of training | `[DATE]` |
| Signature | ____________________ |
| Next refresh due | `[DATE + 1 year]` |

Signed forms are filed in the compliance record system. Loss of the signed acknowledgment is itself a sanction-eligible record-keeping failure.

## 6. Related documents

- [Access control policy](ACCESS_CONTROL.md)
- [Incident response policy](INCIDENT_RESPONSE.md)
- [Sanctions policy](SANCTIONS.md)
- [Technical HIPAA notes](../../HIPAA_NOTES.md)

## 7. Signatures (policy approval)

| Role | Name | Signature | Date |
|---|---|---|---|
| HIPAA Security Officer | `Aimun Malik`, `Founder / HIPAA Security & Privacy Officer` | ____________________ | `[DATE]` |
| HIPAA Privacy Officer | `Aimun Malik`, `Founder / HIPAA Security & Privacy Officer` | ____________________ | `[DATE]` |

Effective date: `[DATE]`. Next review: `[DATE + 1 year]`.
