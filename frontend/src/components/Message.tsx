import clsx from "clsx";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { Source } from "../api/chat";
import { MessageSources } from "./MessageSources";

interface Props {
  role: "user" | "assistant";
  content: string;
  pending?: boolean;
  /** Retrieval citations — rendered as a pill row below the body on
   *  assistant messages. Omitted / empty array = no sources row. */
  sources?: Source[];
}

export function Message({ role, content, pending, sources }: Props) {
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
      {sources && sources.length > 0 && <MessageSources sources={sources} />}
    </div>
  );
}
