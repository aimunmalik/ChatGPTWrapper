export interface AppConfig {
  awsRegion: string;
  cognitoUserPoolId: string;
  cognitoClientId: string;
  cognitoDomain: string;
  apiEndpoint: string;
  redirectUri: string;
  postLogoutRedirectUri: string;
}

function requireEnv(name: string): string {
  const value = import.meta.env[name];
  if (!value) {
    throw new Error(
      `Missing required env var ${name}. Copy .env.example to .env.local and fill it in.`,
    );
  }
  return value as string;
}

const origin = typeof window !== "undefined" ? window.location.origin : "";

export const config: AppConfig = {
  awsRegion: requireEnv("VITE_AWS_REGION"),
  cognitoUserPoolId: requireEnv("VITE_COGNITO_USER_POOL_ID"),
  cognitoClientId: requireEnv("VITE_COGNITO_CLIENT_ID"),
  cognitoDomain: requireEnv("VITE_COGNITO_DOMAIN"),
  apiEndpoint: requireEnv("VITE_API_ENDPOINT").replace(/\/$/, ""),
  redirectUri: `${origin}/callback`,
  postLogoutRedirectUri: origin,
};

export const cognitoIssuer = `https://cognito-idp.${config.awsRegion}.amazonaws.com/${config.cognitoUserPoolId}`;
export const cognitoHostedUi = `https://${config.cognitoDomain}`;
