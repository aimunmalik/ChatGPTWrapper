export interface ModelOption {
  id: string;
  label: string;
  hint: string;
}

export const MODEL_OPTIONS: ModelOption[] = [
  {
    id: "us.anthropic.claude-sonnet-4-6",
    label: "Claude Sonnet 4.6",
    hint: "Balanced — fast and capable. Default.",
  },
  {
    id: "us.anthropic.claude-opus-4-7",
    label: "Claude Opus 4.7",
    hint: "Deepest reasoning. Slower and higher cost.",
  },
  {
    id: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    label: "Claude Haiku 4.5",
    hint: "Fastest. Best for short, simple asks.",
  },
];

export const DEFAULT_MODEL = MODEL_OPTIONS[0].id;

interface Props {
  value: string;
  onChange: (modelId: string) => void;
  disabled?: boolean;
}

export function ModelPicker({ value, onChange, disabled }: Props) {
  const current = MODEL_OPTIONS.find((m) => m.id === value) ?? MODEL_OPTIONS[0];
  return (
    <label className="model-picker">
      <span className="model-picker__label">Model</span>
      <select
        className="model-picker__select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      >
        {MODEL_OPTIONS.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label}
          </option>
        ))}
      </select>
      <span className="model-picker__hint" title={current.hint}>
        {current.hint}
      </span>
    </label>
  );
}
