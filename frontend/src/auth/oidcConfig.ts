import { WebStorageStateStore } from "oidc-client-ts";
import type { AuthProviderProps } from "react-oidc-context";

import { cognitoHostedUi, cognitoIssuer, config } from "../config";

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
};

export function buildLogoutUrl(idTokenHint?: string): string {
  const params = new URLSearchParams({
    client_id: config.cognitoClientId,
    logout_uri: config.postLogoutRedirectUri,
  });
  if (idTokenHint) {
    params.set("id_token_hint", idTokenHint);
  }
  return `${cognitoHostedUi}/logout?${params.toString()}`;
}
