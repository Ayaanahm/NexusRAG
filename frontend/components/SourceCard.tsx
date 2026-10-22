"use client";

import { useState } from "react";
import { toPlain } from "../lib/markdown";
import { AnswerText } from "./AnswerText";
import { RankTrace } from "./RankTrace";

export type SourceView = {
  key: string;
  n: number;
  title: string;
  docId: string;
  url: string | null;
  text: string;
  ranks: { bm25: number | null; dense: number | null; fused: number | null };
  cited?: boolean;
};

export function SourceCard({ s, active }: { s: SourceView; active: boolean }) {
  const [open, setOpen] = useState(false);
  const plain = toPlain(s.text).replace(/\n{2,}/g, "\n");
  const long = plain.length > 320;
  const title = toPlain(s.title) || s.docId;
  return (
    <article id={`source-${s.n}`} className={`source ${active ? "active" : ""} ${s.cited ? "cited" : ""}`}>
      <header>
        <span className="source-n" aria-hidden>
          {s.n}
        </span>
        <div className="source-head">
          <h3>
            {s.url ? (
              <a href={s.url} target="_blank" rel="noreferrer">
                {title}
              </a>
            ) : (
              title
            )}
          </h3>
          <p className="source-doc">{s.docId}</p>
        </div>
      </header>
      <RankTrace {...s.ranks} />
      {open ? (
        <AnswerText text={s.text} sourceCount={0} onCite={() => {}} citations={false} />
      ) : (
        <p className="source-text">{long ? `${plain.slice(0, 320).trimEnd()}…` : plain}</p>
      )}
      {long && (
        <button type="button" className="link-button" onClick={() => setOpen(!open)} aria-expanded={open}>
          {open ? "Show less" : "Show full passage"}
        </button>
      )}
    </article>
  );
}
