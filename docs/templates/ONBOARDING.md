# `<PRODUCT-NAME>` — User Onboarding & Offboarding

> **How to use this doc:** the end-to-end flow for adding, promoting,
> demoting, and removing users. These are the steps the 2-3 people at the
> org who administer this product will actually follow. Keep it concrete
> and avoid "just run Terraform" shortcuts — this doc needs to work at
> 4pm Friday when someone leaves.

**Product:** `<PRODUCT-NAME>`
**Cognito User Pool ID:** `<us-east-1_XXXXXXXXX>` (see Terraform output
`cognito_user_pool_id`)
**Cognito Hosted UI URL:** `<https://<domain>.auth.<region>.amazoncognito.com>`
**Admin group name:** `admins`
**Who can perform user admin:** members of the `admins` Cognito group
with AWS console access.

---

## Role model

| Role | Group membership | Capabilities |
|---|---|---|
| **User** | (none) | Sign in, use the product end-user features |
| **Admin** | `admins` | All user capabilities + admin routes (e.g. KB management) |

There is no "read-only admin" or "billing admin" today. If a future
product needs finer-grained roles, add new Cognito groups and add
corresponding `require_group()` checks server-side.

---

## Scenario A: Add a new user

**When:** new team member needs access.

1. Decide whether they need admin. Default: **no**. Err on
   least-privilege — promotions are easy, revocations are the painful
   part.
2. Open AWS Console → Cognito → User pools → `<pool-id>` → **Users**
   tab.
3. Click **Create user**.
4. Fill in:
   - **Invitation**: "Send an email invitation" (default).
   - **Username**: their work email (this is what they type at login).
   - **Email**: same as username.
   - **Temporary password**: check "Generate a password" unless you're
     sitting next to them.
   - **Mark email as verified**: yes.
5. Click **Create user**.
6. If they need admin: click into the user → **Add user to group** →
   select `admins` → Add.
7. Cognito emails the user a one-time password. They follow the link,
   set a new password, and are prompted to enroll MFA via TOTP
   (Google Authenticator / Authy / 1Password) before they can use
   the product.

**Verify access is working:**
- Confirm with the user that they can sign in and complete one
  primary action.
- Check CloudWatch logs for their first session's events —
  `userId: <their-cognito-sub>`.

**Document it** (e.g. in a shared spreadsheet or wiki):
- Name, email, date added, whether admin, approved by whom.

---

## Scenario B: Promote a user to admin

1. Cognito console → Users → find the user → **Add user to group**.
2. Select `admins`, Add.
3. User needs to **sign out and sign in again** for the new group to
   appear in their JWT. (Group membership is baked into the access
   token at issue time.)
4. Verify: ask them to open the admin surface (e.g. the KB management
   modal in Praxis) — it should appear.

**Document it**: who promoted whom, when, why, approved by whom.

---

## Scenario C: Demote an admin to user

1. Cognito console → Users → find the user → click into their
   group memberships → **Remove from group** → `admins`.
2. **Force sign-out immediately**, so their current access token
   (which still claims admin) becomes invalid:
   ```
   aws cognito-idp admin-user-global-sign-out \
     --user-pool-id <pool-id> --username <username>
   ```
3. Verify: ask them to sign in again and confirm the admin surface
   is gone. Alternatively, check CloudWatch for any `require_admin`
   rejections post-demotion.

---

## Scenario D: User leaves the org (offboarding)

**Complete within 24 hours of their last day** — HIPAA Workforce Security
requires prompt termination of access. The exact steps depend on whether
the product uses Microsoft 365 SSO or local Cognito users.

### D.1 — If the product uses M365 / Entra federation

1. **Disable the user in Microsoft 365 first** (this is the source of
   truth for federated identity):
   - admin.microsoft.com → Users → Active users → find the user → click
     the row → **Block sign-in** toggle → On.
   - Or via PowerShell: `Set-MsolUser -UserPrincipalName <upn> -BlockCredential $true`.
   - This stops any new sign-in to Praxis (or any M365-federated app).
2. **Belt-and-suspenders: disable in Cognito too** so the existing
   refresh token can't be used to silently renew the session in the
   1-day window before it expires:
   ```
   aws cognito-idp admin-disable-user \
     --user-pool-id <pool-id> --username <federated-username>
   aws cognito-idp admin-user-global-sign-out \
     --user-pool-id <pool-id> --username <federated-username>
   ```
   The federated username typically looks like `microsoft_<entra-oid>`
   — list users to find it: `aws cognito-idp list-users --user-pool-id <pool-id>`.
3. If they were a Praxis admin: remove from the `admins` group:
   ```
   aws cognito-idp admin-remove-user-from-group \
     --user-pool-id <pool-id> --username <federated-username> \
     --group-name admins
   ```
4. Continue with steps 5-6 below (audit + artifact recovery).

### D.2 — If the product uses local Cognito users only (or for break-glass accounts)

1. Cognito console → Users → find the user.
2. **Disable** the user (don't delete — you may want audit continuity):
   ```
   aws cognito-idp admin-disable-user \
     --user-pool-id <pool-id> --username <username>
   ```
3. **Force sign-out**:
   ```
   aws cognito-idp admin-user-global-sign-out \
     --user-pool-id <pool-id> --username <username>
   ```
4. If they were an admin: remove from the `admins` group.

### D.3 — Common steps (both paths)

5. **Review what they had access to:**
   - CloudWatch log search for `userId: <their-sub>` in the last 90 days.
   - Any AWS console access? IAM user for AWS? See separate IAM offboarding
     below.
   - Any local devices with product data cached? (Per your device use
     policy — see HIPAA checklist § 2.)
6. Recover any work artifacts they owned (conversation history,
   uploaded documents) per your internal policy.

**Retain the disabled Cognito user for at least `<X>` months** for audit
purposes. Delete only after that retention window.

**Document it**: offboarded date, done by whom, reviewed by whom.

---

## Scenario E: AWS console / IAM user for a developer

**When:** a developer needs direct AWS access for debugging, separate
from product sign-in.

1. Create an IAM user (not a federated login) — smaller blast radius.
2. Attach the `<product>-<env>-developer-read` customer-managed policy
   (read-only CloudWatch / DDB / Lambda metadata). See the Praxis
   `infra/envs/dev/developer_read.tf` for the canonical template.
3. **Do not** grant `AdministratorAccess` or even `PowerUserAccess` to
   human users. Those are for automation (GitHub Actions OIDC role).
4. **Always enable MFA** on the IAM user:
   ```
   aws iam enable-mfa-device --user-name <dev-user> \
     --serial-number <mfa-arn> \
     --authentication-code-1 <code1> \
     --authentication-code-2 <code2>
   ```
5. Issue access keys only when strictly necessary. Prefer AWS SSO /
   Identity Center if the org size justifies setting it up.

**On developer offboarding**: deactivate keys, delete user, within 24h.

---

## Scenario F: Break-glass admin

**Every product should have one break-glass admin account.**

- **Purpose:** recover access if the primary admin loses their MFA
  device or is otherwise locked out.
- **Cognito user**: a real email (`breakglass+<product>@annaautismcare.com`
  or equivalent), in the `admins` group, MFA enrolled.
- **Credentials stored:** in a secure offline/physical location, NOT
  in a shared password manager. Access to the credentials logged.
- **Used only for recovery.** Every use triggers a post-use password
  reset + MFA re-enrollment.
- **Audit quarterly**: the credentials still work, MFA token still
  generates valid codes, only the expected people have access to the
  physical storage.

Document where the credentials live without revealing them:
- Primary storage location: `<physical safe / lockbox / ...>`
- Who has access to the physical storage: `<names>`
- Last verified accessible: `<date>`

---

## Scenario G: Reset a user's password

**When:** user forgot their password and the Hosted UI "Forgot
password" flow isn't working for them (unusual).

1. Cognito console → Users → find the user → **Reset password** →
   confirm.
2. A new temporary password is emailed to them.
3. They sign in, get prompted to set a new permanent password.
4. MFA is preserved unless you explicitly reset it too.

---

## Scenario H: Reset a user's MFA

**When:** user lost their MFA device.

1. **Verify identity out of band** — a phone call (ideally video).
   This is a social engineering vector. A real person asking in person
   or on video is fine. An email "I lost my MFA, can you reset it?"
   is not.
2. Cognito console → Users → find the user → scroll to **Multi-factor
   authentication** → **Reset MFA**.
3. User signs in with their password; Cognito prompts them to
   re-enroll MFA.
4. **Record the event** in the user audit log — who requested, who
   verified, when.

---

## Scenario I: Access review (quarterly)

Every quarter, run this:

1. Export the full Cognito user list:
   ```
   aws cognito-idp list-users --user-pool-id <pool-id> \
     --max-results 60 --output json > users-$(date +%Y%m%d).json
   ```
2. Export the admin group membership:
   ```
   aws cognito-idp list-users-in-group --user-pool-id <pool-id> \
     --group-name admins --output json > admins-$(date +%Y%m%d).json
   ```
3. For each user: is this person still on the team? Do they still need
   this product?
4. For each admin: do they still need admin (vs. user)?
5. Disable anyone who shouldn't have access anymore (Scenario D).
6. Demote anyone who shouldn't still be admin (Scenario C).
7. **Sign and date** the review (store in `docs/access-reviews/`).

---

## A note on usernames

- Cognito usernames in this setup are **email addresses** (the
  product uses email-alias-as-identifier).
- Changing a username is not supported. If a user's email changes,
  create a new Cognito user with the new email, migrate any
  product-level data by `userId` (Cognito `sub`, which is stable),
  and disable the old user.
- `sub` is the stable identifier — always use it in application logs,
  DDB keys, etc. Never use the email as a primary key.

---

## Emergency contacts

Keep these current:

- **Primary admin**: `<name>` — `<email>` — `<phone>`
- **HIPAA Security Officer**: Aimun Malik — aimun@annaautismcare.com
- **AWS account owner / root**: `<name>` — `<email>`
- **Break-glass physical storage access**: `<location + custodian>`
