import { useRef } from "react";

const ACCEPT =
  "application/pdf,image/png,image/jpeg,.xlsx,.xls,.docx,.csv,.txt,text/plain";
const MAX_SIZE_BYTES = 50 * 1024 * 1024;

interface Props {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
}

export function AttachmentPicker({ onFiles, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(e.target.files ?? []);
    const accepted: File[] = [];
    for (const f of picked) {
      if (f.size > MAX_SIZE_BYTES) {
        const mb = (f.size / (1024 * 1024)).toFixed(1);
        alert(`File ${f.name} is ${mb}MB; 50MB limit`);
        continue;
      }
      accepted.push(f);
    }
    // Reset the input so picking the same file again re-triggers onChange.
    e.target.value = "";
    if (accepted.length > 0) onFiles(accepted);
  }

  return (
    <>
      <button
        type="button"
        className="btn btn--icon chat__attach-btn"
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
        aria-label="Attach files"
        title="Attach files"
      >
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#FF7896"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
        </svg>
        <span className="chat__attach-label">Attach</span>
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPT}
        style={{ display: "none" }}
        onChange={handleChange}
      />
    </>
  );
}
