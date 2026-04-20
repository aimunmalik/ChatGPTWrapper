import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

export interface Command {
  id: string;
  label: string;
  group: string;
  hint?: string;
  keywords?: string;
  onRun: () => void;
}

interface Props {
  open: boolean;
  commands: Command[];
  onClose: () => void;
}

function fuzzyMatch(query: string, text: string): boolean {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  if (!q) return true;
  let ti = 0;
  for (const ch of q) {
    ti = t.indexOf(ch, ti);
    if (ti === -1) return false;
    ti += 1;
  }
  return true;
}

export function CommandPalette({ open, commands, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
      queueMicrotask(() => inputRef.current?.focus());
    }
  }, [open]);

  const filtered = useMemo(() => {
    if (!query.trim()) return commands;
    return commands.filter((c) =>
      fuzzyMatch(query, `${c.label} ${c.group} ${c.keywords ?? ""}`),
    );
  }, [query, commands]);

  useEffect(() => {
    setActive(0);
  }, [query]);

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(i + 1, filtered.length - 1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      const cmd = filtered[active];
      if (cmd) {
        cmd.onRun();
        onClose();
      }
    }
  }

  if (!open) return null;

  const grouped = filtered.reduce<Record<string, Command[]>>((acc, cmd) => {
    (acc[cmd.group] ??= []).push(cmd);
    return acc;
  }, {});

  let idx = -1;

  return (
    <div
      className="cmdk-overlay"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="cmdk-panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Command palette"
      >
        <input
          ref={inputRef}
          className="cmdk-input"
          placeholder="Type a command or search…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        {filtered.length === 0 ? (
          <div className="cmdk-empty">No matching commands.</div>
        ) : (
          <ul className="cmdk-list">
            {Object.entries(grouped).map(([group, items]) => (
              <li key={group}>
                <div className="cmdk-group-label">{group}</div>
                <ul className="cmdk-list" style={{ padding: 0, maxHeight: "none" }}>
                  {items.map((cmd) => {
                    idx += 1;
                    const isActive = idx === active;
                    return (
                      <li
                        key={cmd.id}
                        className={clsx("cmdk-item", isActive && "cmdk-item--active")}
                        onMouseEnter={() => setActive(idx)}
                        onClick={() => {
                          cmd.onRun();
                          onClose();
                        }}
                      >
                        <span className="cmdk-item__label">{cmd.label}</span>
                        {cmd.hint && <span className="cmdk-item__hint">{cmd.hint}</span>}
                      </li>
                    );
                  })}
                </ul>
              </li>
            ))}
          </ul>
        )}
        <div className="cmdk-footer">
          <span>
            <span className="kbd">↑</span> <span className="kbd">↓</span> navigate{" "}
            <span className="kbd">↵</span> select
          </span>
          <span>
            <span className="kbd">Esc</span> close
          </span>
        </div>
      </div>
    </div>
  );
}
