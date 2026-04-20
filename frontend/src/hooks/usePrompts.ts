import { useCallback, useEffect, useRef, useState } from "react";

import {
  createPrompt,
  deletePrompt,
  listPrompts,
  updatePrompt,
} from "../api/prompts";
import type { Prompt, PromptInput } from "../api/prompts";

interface Options {
  accessToken: string;
}

interface UsePrompts {
  prompts: Prompt[];
  isLoading: boolean;
  error: string | null;
  create: (input: PromptInput) => Promise<Prompt>;
  update: (promptId: string, input: PromptInput) => Promise<Prompt>;
  remove: (promptId: string) => Promise<void>;
  refresh: () => Promise<void>;
}

/** Sort newest-first by updatedAt so recently-edited prompts surface at the
 *  top of both the palette "My prompts" group and the library modal. */
function sortPrompts(list: Prompt[]): Prompt[] {
  return [...list].sort((a, b) => b.updatedAt - a.updatedAt);
}

export function usePrompts({ accessToken }: Options): UsePrompts {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tokenRef = useRef(accessToken);
  useEffect(() => {
    tokenRef.current = accessToken;
  }, [accessToken]);

  const refresh = useCallback(async () => {
    const token = tokenRef.current;
    if (!token) {
      setPrompts([]);
      return;
    }
    try {
      const resp = await listPrompts(token);
      setPrompts(sortPrompts(resp.prompts));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load prompts");
    }
  }, []);

  // Initial load whenever the access token changes.
  useEffect(() => {
    let cancelled = false;
    if (!accessToken) {
      setPrompts([]);
      setIsLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setIsLoading(true);
    setError(null);
    listPrompts(accessToken)
      .then((resp) => {
        if (!cancelled) setPrompts(sortPrompts(resp.prompts));
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load prompts");
          setPrompts([]);
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  const create = useCallback(async (input: PromptInput): Promise<Prompt> => {
    const token = tokenRef.current;
    if (!token) throw new Error("Not authenticated");
    // Optimistic: insert a placeholder with a temporary id until the server
    // echoes back the real one.
    const tempId = `local-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const now = Date.now();
    const optimistic: Prompt = {
      promptId: tempId,
      title: input.title,
      body: input.body,
      createdAt: now,
      updatedAt: now,
    };
    setPrompts((prev) => sortPrompts([optimistic, ...prev]));
    try {
      const resp = await createPrompt(token, input);
      setPrompts((prev) =>
        sortPrompts(prev.map((p) => (p.promptId === tempId ? resp.prompt : p))),
      );
      return resp.prompt;
    } catch (err) {
      // Rollback
      setPrompts((prev) => prev.filter((p) => p.promptId !== tempId));
      setError(err instanceof Error ? err.message : "Failed to create prompt");
      throw err;
    }
  }, []);

  const update = useCallback(
    async (promptId: string, input: PromptInput): Promise<Prompt> => {
      const token = tokenRef.current;
      if (!token) throw new Error("Not authenticated");
      const snapshot = prompts;
      const now = Date.now();
      setPrompts((prev) =>
        sortPrompts(
          prev.map((p) =>
            p.promptId === promptId
              ? { ...p, title: input.title, body: input.body, updatedAt: now }
              : p,
          ),
        ),
      );
      try {
        const resp = await updatePrompt(token, promptId, input);
        setPrompts((prev) =>
          sortPrompts(prev.map((p) => (p.promptId === promptId ? resp.prompt : p))),
        );
        return resp.prompt;
      } catch (err) {
        setPrompts(snapshot);
        setError(err instanceof Error ? err.message : "Failed to update prompt");
        throw err;
      }
    },
    [prompts],
  );

  const remove = useCallback(
    async (promptId: string): Promise<void> => {
      const token = tokenRef.current;
      if (!token) throw new Error("Not authenticated");
      const snapshot = prompts;
      setPrompts((prev) => prev.filter((p) => p.promptId !== promptId));
      try {
        await deletePrompt(token, promptId);
      } catch (err) {
        setPrompts(snapshot);
        setError(err instanceof Error ? err.message : "Failed to delete prompt");
        throw err;
      }
    },
    [prompts],
  );

  return { prompts, isLoading, error, create, update, remove, refresh };
}
