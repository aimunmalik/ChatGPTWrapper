import clsx from "clsx";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  role: "user" | "assistant";
  content: string;
  pending?: boolean;
}

export function Message({ role, content, pending }: Props) {
  return (
    <div className={clsx("message", `message--${role}`)}>
      <div className="message__role">{role === "user" ? "You" : "Praxis"}</div>
      <div className="message__body">
        {pending ? (
          <span className="message__dots" aria-label="Thinking">
            <span /> <span /> <span />
          </span>
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        )}
      </div>
    </div>
  );
}
