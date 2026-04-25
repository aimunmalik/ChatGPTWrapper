import { WebStorageStateStore } from "oidc-client-ts";
import type { AuthProviderProps } from "react-oidc-context";

import { cognitoHostedUi, cognitoIssuer, config } from "../config";

// When VITE_DEFAULT_IDP is set, oidc-client-ts forwards `identity_provider=<name>`
// on the /oauth2/authorize redirect. Cognito then bypasses its picker UI and
// bounces straight to that federated IdP (e.g. Microsoft Entra). Leave unset
// to keep the picker visible — useful while local username/password is still
// the break-glass path.
const extraQueryParams = config.defaultIdp
  ? { identity_provider: config.defaultIdp }
  : undefined;

export const oidcConfig: AuthProviderProps = {
  authority: cognitoIssuer,
  metadata: {
    issuer: cognitoIssuer,
    authorization_endpoint: `${cognitoHostedUi}/oauth2/authorize`,
    token_endpoint: `${cognitoHostedUi}/oauth2/token`,
    userinfo_endpoint: `${cognitoHostedUi}/oauth2/userInfo`,
    end_session_endpoint: `${cognitoHostedUi}/logout`,
    jwks_uri: `${cognitoIssuer}/.well-known/jwks.json`,
    revocation_endpoint: `${cognitoHostedUi}/oauth2/revoke`,
  },
  client_id: config.cognitoClientId,
  redirect_uri: config.redirectUri,
  post_logout_redirect_uri: config.postLogoutRedirectUri,
  response_type: "code",
  scope: "openid email profile",
  loadUserInfo: false,
  automaticSilentRenew: true,
  userStore: new WebStorageStateStore({ store: window.localStorage }),
  extraQueryParams,
};

export function buildLogoutUrl(idTokenHint?: string): string {
  // Land back on /login?signedout=1 instead of root. The login page checks
  // that flag and shows an explicit "Sign in" button instead of silently
  // re-initiating the sign-in flow — otherwise the user gets bounced
  // straight back into the app via SSO (their M365 session is still alive)
  // and "sign out" appears to do nothing.
  const params = new URLSearchParams({
    client_id: config.cognitoClientId,
    logout_uri: `${config.postLogoutRedirectUri}/login?signedout=1`,
  });
  if (idTokenHint) {
    params.set("id_token_hint", idTokenHint);
  }
  return `${cognitoHostedUi}/logout?${params.toString()}`;
}
