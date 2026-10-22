import { Fragment } from "react";
import { type Inline, parseMarkdown } from "../lib/markdown";

type Props = {
  text: string;
  sourceCount: number;
  onCite: (n: number) => void;
  citations?: boolean;
};

function renderInline(inl: Inline[], sourceCount: number, onCite: (n: number) => void, citations = true) {
  return inl.map((x, i) => {
    switch (x.t) {
      case "text":
        return <Fragment key={i}>{x.v}</Fragment>;
      case "code":
        return <code key={i}>{x.v}</code>;
      case "strong":
        return <strong key={i}>{x.v}</strong>;
      case "link":
        return (
          <a key={i} href={x.href} target="_blank" rel="noreferrer">
            {x.v}
          </a>
        );
      case "cite":
        if (!citations) return <Fragment key={i}>[{x.n.join(", ")}]</Fragment>;
        return (
          <span key={i} className="cites">
            {x.n.map((n) =>
              n >= 1 && n <= sourceCount ? (
                <button key={n} type="button" className="cite" onClick={() => onCite(n)} aria-label={`Show source ${n}`}>
                  {n}
                </button>
              ) : (
                <span key={n} className="cite invalid" title="This number does not match any retrieved source">
                  {n}?
                </span>
              ),
            )}
          </span>
        );
    }
  });
}

export function AnswerText({ text, sourceCount, onCite, citations = true }: Props) {
  return (
    <div className={citations ? "answer-text" : "passage-text"}>
      {parseMarkdown(text).map((b, i) => {
        if (b.t === "pre")
          return (
            <pre key={i}>
              <code>{b.code}</code>
            </pre>
          );
        if (b.t === "p") return <p key={i}>{renderInline(b.inl, sourceCount, onCite, citations)}</p>;
        const List = b.t;
        return (
          <List key={i}>
            {b.items.map((it, j) => (
              <li key={j}>{renderInline(it, sourceCount, onCite, citations)}</li>
            ))}
          </List>
        );
      })}
    </div>
  );
}
