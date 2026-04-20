import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

import type { Prompt, PromptInput } from "../api/prompts";
import type { PromptTemplate } from "../prompts";

interface Props {
  open: boolean;
  onClose: () => void;
  prompts: Prompt[];
  onCreate: (input: PromptInput) => Promise<Prompt>;
  onUpdate: (promptId: string, input: PromptInput) => Promise<Prompt>;
  onDelete: (promptId: string) => Promise<void>;
  starterTemplates: PromptTemplate[];
}

type Tab = "mine" | "starter";

const TITLE_MAX = 120;
const BODY_MAX = 20000;

interface DraftState {
  /** `null` = not editing, `""` = editing a new prompt, string = editing by id. */
  editingId: string | null | "";
  title: string;
  body: string;
}

const EMPTY_DRAFT: DraftState = { editingId: null, title: "", body: "" };

export function PromptLibrary({
  open,
  onClose,
  prompts,
  onCreate,
  onUpdate,
  onDelete,
  starterTemplates,
}: Props) {
  const [tab, setTab] = useState<Tab>("mine");
  const [draft, setDraft] = useState<DraftState>(EMPTY_DRAFT);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const titleInputRef = useRef<HTMLInputElement | null>(null);

  // Reset everything whenever the modal closes so next open is a clean slate.
  useEffect(() => {
    if (!open) {
      setTab("mine");
      setDraft(EMPTY_DRAFT);
      setFormError(null);
      setSaving(false);
    }
  }, [open]);

  // Focus the title field whenever we enter editing/creation mode.
  useEffect(() => {
    if (draft.editingId !== null) {
      queueMicrotask(() => titleInputRef.current?.focus());
    }
  }, [draft.editingId]);

  // Esc closes the modal — but only if we're not actively editing, so a
  // mid-typing Esc doesn't nuke the draft.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && draft.editingId === null) {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose, draft.editingId]);

  const editingExisting = useMemo(
    () =>
      draft.editingId !== null && draft.editingId !== ""
        ? prompts.find((p) => p.promptId === draft.editingId) ?? null
        : null,
    [draft.editingId, prompts],
  );

  function beginNew(prefill?: { title?: string; body?: string }) {
    setTab("mine");
    setFormError(null);
    setDraft({
      editingId: "",
      title: prefill?.title ?? "",
      body: prefill?.body ?? "",
    });
  }

  function beginEdit(p: Prompt) {
    setFormError(null);
    setDraft({ editingId: p.promptId, title: p.title, body: p.body });
  }

  function cancelDraft() {
    setDraft(EMPTY_DRAFT);
    setFormError(null);
  }

  async function saveDraft() {
    const title = draft.title.trim();
    const body = draft.body.trim();
    if (!title) {
      setFormError("Title is required.");
      return;
    }
    if (title.length > TITLE_MAX) {
      setFormError(`Title must be ${TITLE_MAX} characters or fewer.`);
      return;
    }
    if (!body) {
      setFormError("Body is required.");
      return;
    }
    if (body.length > BODY_MAX) {
      setFormError(`Body must be ${BODY_MAX} characters or fewer.`);
      return;
    }
    setSaving(true);
    setFormError(null);
    try {
      if (editingExisting) {
        await onUpdate(editingExisting.promptId, { title, body });
      } else {
        await onCreate({ title, body });
      }
      setDraft(EMPTY_DRAFT);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : "Save failed.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!editingExisting) return;
    if (!window.confirm(`Delete "${editingExisting.title}"? This can't be undone.`)) {
      return;
    }
    setSaving(true);
    setFormError(null);
    try {
      await onDelete(editingExisting.promptId);
      setDraft(EMPTY_DRAFT);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : "Delete failed.");
    } finally {
      setSaving(false);
    }
  }

  if (!open) return null;

  return (
    <div
      className="cmdk-overlay prompt-library"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="cmdk-panel prompt-library__panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Prompt library"
        aria-modal="true"
      >
        <div className="prompt-library__header">
          <h2 className="prompt-library__title">Prompt library</h2>
          <button
            type="button"
            className="prompt-library__close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="prompt-library__tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "mine"}
            className={clsx(
              "prompt-library__tab",
              tab === "mine" && "prompt-library__tab--active",
            )}
            onClick={() => setTab("mine")}
          >
            My prompts
            <span className="prompt-library__tab-count">{prompts.length}</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "starter"}
            className={clsx(
              "prompt-library__tab",
              tab === "starter" && "prompt-library__tab--active",
            )}
            onClick={() => setTab("starter")}
          >
            Starter templates
            <span className="prompt-library__tab-count">
              {starterTemplates.length}
            </span>
          </button>
        </div>

        <div className="prompt-library__body">
          {tab === "mine" ? (
            <MyPromptsTab
              prompts={prompts}
              draft={draft}
              editingExisting={editingExisting}
              saving={saving}
              formError={formError}
              titleInputRef={titleInputRef}
              onBeginNew={() => beginNew()}
              onBeginEdit={beginEdit}
              onChangeTitle={(title) => setDraft((d) => ({ ...d, title }))}
              onChangeBody={(body) => setDraft((d) => ({ ...d, body }))}
              onSave={saveDraft}
              onCancel={cancelDraft}
              onDelete={handleDelete}
            />
          ) : (
            <StarterTab
              templates={starterTemplates}
              onCopy={(t) => beginNew({ title: t.label, body: t.template })}
            />
          )}
        </div>
      </div>
    </div>
  );
}

interface MyPromptsTabProps {
  prompts: Prompt[];
  draft: DraftState;
  editingExisting: Prompt | null;
  saving: boolean;
  formError: string | null;
  titleInputRef: React.MutableRefObject<HTMLInputElement | null>;
  onBeginNew: () => void;
  onBeginEdit: (p: Prompt) => void;
  onChangeTitle: (title: string) => void;
  onChangeBody: (body: string) => void;
  onSave: () => void;
  onCancel: () => void;
  onDelete: () => void;
}

function MyPromptsTab({
  prompts,
  draft,
  editingExisting,
  saving,
  formError,
  titleInputRef,
  onBeginNew,
  onBeginEdit,
  onChangeTitle,
  onChangeBody,
  onSave,
  onCancel,
  onDelete,
}: MyPromptsTabProps) {
  const editing = draft.editingId !== null;

  if (editing) {
    return (
      <form
        className="prompt-library__form"
        onSubmit={(e) => {
          e.preventDefault();
          onSave();
        }}
      >
        <label className="prompt-library__field">
          <span className="prompt-library__field-label">Title</span>
          <input
            ref={titleInputRef}
            type="text"
            className="prompt-library__input"
            maxLength={TITLE_MAX}
            value={draft.title}
            onChange={(e) => onChangeTitle(e.target.value)}
            placeholder="e.g. Draft BIP for aggressive behaviors"
            disabled={saving}
          />
          <span className="prompt-library__field-hint">
            {draft.title.length}/{TITLE_MAX}
          </span>
        </label>
        <label className="prompt-library__field">
          <span className="prompt-library__field-label">Body</span>
          <textarea
            className="prompt-library__textarea"
            rows={12}
            maxLength={BODY_MAX}
            value={draft.body}
            onChange={(e) => onChangeBody(e.target.value)}
            placeholder="Template text. Use brackets like [client age] for fields you'll fill in each time."
            disabled={saving}
          />
          <span className="prompt-library__field-hint">
            {draft.body.length}/{BODY_MAX}
          </span>
        </label>
        {formError && <div className="prompt-library__error">{formError}</div>}
        <div className="prompt-library__actions">
          {editingExisting && (
            <button
              type="button"
              className="btn prompt-library__btn-danger"
              onClick={onDelete}
              disabled={saving}
            >
              Delete
            </button>
          )}
          <span className="prompt-library__actions-spacer" />
          <button
            type="button"
            className="btn"
            onClick={onCancel}
            disabled={saving}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn btn--primary"
            disabled={saving}
          >
            {saving ? "Saving…" : editingExisting ? "Save changes" : "Save prompt"}
          </button>
        </div>
      </form>
    );
  }

  return (
    <>
      <div className="prompt-library__toolbar">
        <button type="button" className="btn btn--primary" onClick={onBeginNew}>
          New prompt
        </button>
      </div>
      {prompts.length === 0 ? (
        <div className="prompt-library__empty">
          You haven't saved any prompts yet. Starter templates are available
          under Quick prompts in ⌘K.
        </div>
      ) : (
        <ul className="prompt-library__list">
          {prompts.map((p) => (
            <li key={p.promptId} className="prompt-library__item">
              <button
                type="button"
                className="prompt-library__item-main"
                onClick={() => onBeginEdit(p)}
              >
                <span className="prompt-library__item-title">{p.title}</span>
                <span className="prompt-library__item-preview">
                  {previewLine(p.body)}
                </span>
              </button>
              <button
                type="button"
                className="prompt-library__item-edit"
                onClick={() => onBeginEdit(p)}
                aria-label={`Edit ${p.title}`}
              >
                Edit
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

interface StarterTabProps {
  templates: PromptTemplate[];
  onCopy: (t: PromptTemplate) => void;
}

function StarterTab({ templates, onCopy }: StarterTabProps) {
  return (
    <>
      <div className="prompt-library__hint">
        Starter templates are built in and read-only. Copy any one into your
        library to customize.
      </div>
      <ul className="prompt-library__list">
        {templates.map((t) => (
          <li key={t.id} className="prompt-library__item">
            <div className="prompt-library__item-main prompt-library__item-main--static">
              <span className="prompt-library__item-title">{t.label}</span>
              <span className="prompt-library__item-preview">{t.hint}</span>
            </div>
            <button
              type="button"
              className="btn prompt-library__item-copy"
              onClick={() => onCopy(t)}
            >
              Copy to my prompts
            </button>
          </li>
        ))}
      </ul>
    </>
  );
}

function previewLine(body: string): string {
  const firstNonEmpty = body.split("\n").find((l) => l.trim().length > 0) ?? "";
  const trimmed = firstNonEmpty.trim();
  return trimmed.length > 120 ? `${trimmed.slice(0, 117)}…` : trimmed;
}
