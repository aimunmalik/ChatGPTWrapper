# Microsoft 365 SSO setup (Entra ID federation)

> Wires `praxis.annaautismcare.com` sign-in to ANNA's Microsoft 365 tenant via
> OIDC federation. After this is in place: clinicians click "Sign in with
> Microsoft" → bounce to login.microsoftonline.com → MFA enforced by Entra
> Conditional Access → bounce back to Praxis as a logged-in user.
>
> Replaces (or runs alongside) the local Cognito username + TOTP path.
> The local path stays available as a break-glass route until you choose
> to disable it.

**Prereqs:**
- M365 license that includes Entra ID P1 (Business Premium, E3, E5, or
  standalone Entra ID P1). Business Standard does **not** include this.
- Tenant admin access to admin.microsoft.com (or entra.microsoft.com).
- A Cognito break-glass user already created — see `docs/templates/ONBOARDING.md`
  scenario "Break-glass admin." Do this BEFORE step 1 below; if anything
  goes sideways during cutover, you don't want to be locked out.

---

## Part 1 — Microsoft side (Entra app registration)

### 1.1 Create the app registration

1. Go to <https://entra.microsoft.com> → sign in as a tenant admin.
2. Left nav: **Applications** → **App registrations** → **+ New registration**.
3. Fill in:
   - **Name:** `Praxis - ANNA Health`
   - **Supported account types:** *Accounts in this organizational directory only (Single tenant)*
   - **Redirect URI:**
     - Platform: **Web**
     - URL: `https://anna-chat-dev-<your-suffix>.auth.us-east-1.amazoncognito.com/oauth2/idpresponse`
       - Find `<your-suffix>` from `terraform output cognito.hosted_ui_url`
       - or just look at the existing `cognito_domain_suffix` in `terraform.tfvars`
4. Click **Register**.

### 1.2 Capture the IDs

On the app's **Overview** page, copy these into a scratch file — you'll
paste them into `terraform.tfvars`:

| Label in Entra | Goes into tfvars as |
|---|---|
| **Directory (tenant) ID** | `entra_tenant_id` |
| **Application (client) ID** | `entra_client_id` |

### 1.3 Create a client secret

1. In the app's left nav: **Certificates & secrets** → **Client secrets** → **+ New client secret**.
2. Description: `Cognito federation`
3. Expires: **24 months** (longest option). Set a calendar reminder for
   ~22 months out to rotate before expiry.
4. Click **Add**.
5. **Immediately copy the `Value` column** (NOT the `Secret ID` column) —
   this is the only time it's visible. It looks like
   `Abc1~Defg2hij3KLMn4OPqRsTu5vw6Xyz789ABCdef`.
6. Paste it into `terraform.tfvars` as `entra_client_secret`.

### 1.4 Configure ID token claims

Cognito needs `email` and `name` claims to populate the federated user.
By default Entra includes `name` but `email` is gated.

1. In the app's left nav: **Token configuration** → **+ Add optional claim**.
2. Token type: **ID** → check **email** → click **Add**.
3. If a popup appears about Microsoft Graph permissions, click "Yes" to
   add the required `email` permission.
4. Repeat for **upn** (used as a fallback if email isn't populated).

### 1.5 Configure API permissions (one-time)

The app needs three Graph permissions to read enough about the user to
issue a useful ID token. Most of these are added automatically; verify:

1. Left nav: **API permissions**.
2. You should see (under Microsoft Graph → Delegated):
   - `openid`
   - `profile`
   - `email`
   - `User.Read` (auto-added by Entra)
3. If any are missing: **+ Add a permission** → **Microsoft Graph** →
   **Delegated permissions** → check the missing ones → **Add permissions**.
4. Click **Grant admin consent for ANNA Health** at the top of the
   permissions list. (Only a Global Admin can do this.) The "Status"
   column should turn green for all four.

### 1.6 (Optional but recommended) Restrict to specific users/groups

If you want only certain people to be able to sign in to Praxis even
though they're all in the M365 tenant:

1. Left nav: switch to **Enterprise applications** at the tenant level
   (admin.microsoft.com → Identity → Enterprise applications).
2. Find `Praxis - ANNA Health` → **Properties** → set **Assignment
   required?** to **Yes** → **Save**.
3. Left nav (still on the enterprise app): **Users and groups** → **+ Add
   user/group** → assign individuals or a group like "Praxis Users".

Without this, **every** user in the M365 tenant can sign in to Praxis
(they'll appear in Cognito only on first sign-in). For a small clinical
team that's usually fine; for a larger org you want assignment-required.

### 1.7 (Optional) Configure Conditional Access

This is where you enforce MFA at the Entra level. If your tenant already
has a baseline Conditional Access policy that requires MFA on all apps,
Praxis is automatically covered. If not:

1. <https://entra.microsoft.com> → **Protection** → **Conditional Access**
   → **+ New policy**.
2. Name: `Praxis - require MFA`
3. Users: include **Praxis Users** (or All users).
4. Target resources: select **Cloud apps** → **Select** → search
   `Praxis - ANNA Health`.
5. Grant: **Require multi-factor authentication**.
6. Enable policy: **On** → **Create**.

After this, every Praxis sign-in requires MFA via whatever methods Entra
allows (Authenticator app push, FIDO2, Windows Hello, SMS — your call).

---

## Part 2 — AWS / Praxis side

### 2.1 Fill in tfvars

Open `infra/envs/dev/terraform.tfvars` (gitignored — never commit) and
add the three values you captured in Part 1:

```hcl
entra_tenant_id     = "<the Directory (tenant) ID>"
entra_client_id     = "<the Application (client) ID>"
entra_client_secret = "<the secret Value from step 1.3>"
```

### 2.2 Deploy

Push to main, watch CI deploy. The terraform changes:

- Create `aws_cognito_identity_provider.entra[0]` (OIDC IdP pointed at
  your Entra tenant).
- Update the SPA app client's `supported_identity_providers` to include
  `Microsoft` alongside `COGNITO`.

After deploy, the Cognito Hosted UI sign-in page will show:

- **"Continue with Microsoft"** button (federated path)
- **"Sign in"** form below (local username/password, the break-glass path)

### 2.3 First federated sign-in

1. Open `https://praxis.annaautismcare.com` in an incognito/private window.
2. Click **Continue with Microsoft**.
3. You'll bounce to `login.microsoftonline.com`, sign in with your
   `you@annaautismcare.com` account, complete MFA per your CA policy.
4. You bounce back to Praxis. Cognito has provisioned a new federated
   user under the hood — but they're not in the `admins` group yet, so
   you can chat but can't open the KB modal.

### 2.4 Promote the federated user to admin

You'll have two Cognito users now: your old local one (admin) and a
new federated one. Promote the federated one:

```powershell
$pool = (aws cognito-idp list-user-pools --max-results 10 --query "UserPools[?starts_with(Name, 'anna-chat-dev')].Id" --output text)

# Find the federated user (will have a sub like `microsoft_<entra-oid>`)
aws cognito-idp list-users --user-pool-id $pool --query "Users[*].Username" --output text

# Add to admins group (replace <username> with the federated user's Username)
aws cognito-idp admin-add-user-to-group --user-pool-id $pool --username "<username>" --group-name admins
```

Refresh the app, your federated user should now see the "Manage
knowledge base…" command palette entry.

### 2.5 Verify break-glass still works

Sign out, sign back in via the **local username/password** form (NOT
the Microsoft button). You should still be able to sign in with your
original Cognito user — this proves the break-glass path is intact.

### 2.6 (Optional, do later) Skip the Cognito picker page

Once you're confident SSO works for everyone and your break-glass user
is well-documented and tested:

1. Add `VITE_DEFAULT_IDP=Microsoft` as a GitHub Actions Variable for
   the deploy workflow.
2. Update `.github/workflows/deploy-dev.yml` Build frontend step to pass
   it through:

   ```yaml
   - name: Build frontend
     env:
       …existing vars…
       VITE_DEFAULT_IDP: ${{ vars.VITE_DEFAULT_IDP }}
   ```
3. Push. Next sign-in skips the Cognito picker and goes straight to
   Microsoft. Break-glass users hit the Hosted UI URL directly with
   `?identity_provider=COGNITO` appended to bypass.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `redirect_uri_mismatch` from Microsoft | Entra app registration redirect URI doesn't exactly match Cognito's `idpresponse` URL | Recheck Part 1 step 1.3. Must include `/oauth2/idpresponse` exactly. |
| Sign-in works but Praxis user is missing email | `email` claim not in ID token | Part 1 step 1.4 — add `email` to optional ID token claims. |
| `AADSTS50105: assignment required` | Enterprise app has assignment-required ON but your user isn't assigned | Add yourself in Enterprise applications → Users and groups, OR turn assignment-required OFF. |
| `AADSTS65001: consent_required` | Admin consent not granted | Part 1 step 1.5 — Grant admin consent. |
| Federated user can't open KB | Federated user not in `admins` group | Run the `admin-add-user-to-group` command from step 2.4. |
| `error=invalid_request&error_description=Bad+Identity+Provider+Microsoft` from Cognito | The IdP name in `supported_identity_providers` doesn't match the IdP resource's `provider_name` | Both default to "Microsoft"; if you renamed, keep them in sync. |
| Client secret expired | Entra rotates / expiry hit | Generate new secret in Entra (Part 1 step 1.3) → update `entra_client_secret` in tfvars → re-apply. |

---

## Rotation policy

- **Entra client secret:** rotate every 12-24 months. Calendar reminder
  required at issuance time. Process: generate new secret in Entra,
  update tfvars, terraform apply, then delete the old secret in Entra.
- **No automated rotation** — Cognito doesn't natively poll for new
  secrets. Manual is fine for one app.
