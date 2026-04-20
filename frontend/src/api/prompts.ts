import { apiFetch } from "./client";

/** A user-authored prompt template stored server-side. Matches the DDB shape
 *  defined in docs/PROMPTS_CONTRACT.md. */
export interface Prompt {
  promptId: string;
  title: string;
  body: string;
  createdAt: number;
  updatedAt: number;
}

/** The mutable fields accepted by POST /prompts and PUT /prompts/{id}. Both
 *  endpoints take the same shape — full-replace semantics on update. */
export interface PromptInput {
  title: string;
  body: string;
}

export function listPrompts(
  accessToken: string,
): Promise<{ prompts: Prompt[] }> {
  return apiFetch("/prompts", accessToken);
}

export function createPrompt(
  accessToken: string,
  input: PromptInput,
): Promise<{ prompt: Prompt }> {
  return apiFetch("/prompts", accessToken, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function updatePrompt(
  accessToken: string,
  promptId: string,
  input: PromptInput,
): Promise<{ prompt: Prompt }> {
  return apiFetch(`/prompts/${encodeURIComponent(promptId)}`, accessToken, {
    method: "PUT",
    body: JSON.stringify(input),
  });
}

export function deletePrompt(
  accessToken: string,
  promptId: string,
): Promise<{ deleted: boolean }> {
  return apiFetch(`/prompts/${encodeURIComponent(promptId)}`, accessToken, {
    method: "DELETE",
  });
}
