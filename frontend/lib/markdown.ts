// A deliberately small Markdown subset for model answers: paragraphs, bullet
// and numbered lists, fenced code, inline code, bold, links and [n] citations.
// It produces a token tree, never HTML strings, so model output cannot inject markup.

export type Inline =
  | { t: "text"; v: string }
  | { t: "code"; v: string }
  | { t: "strong"; v: string }
  | { t: "link"; v: string; href: string }
  | { t: "cite"; n: number[] };

export type Block =
  | { t: "p"; inl: Inline[] }
  | { t: "ul" | "ol"; items: Inline[][] }
  | { t: "pre"; lang: string; code: string };

const INLINE_RE = /`([^`]+)`|\*\*([^*]+)\*\*|\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)|\[(\d+(?:\s*,\s*\d+)*)\]/g;

export function parseInline(s: string): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const m of s.matchAll(INLINE_RE)) {
    const i = m.index ?? 0;
    if (i > last) out.push({ t: "text", v: s.slice(last, i) });
    if (m[1] !== undefined) out.push({ t: "code", v: m[1] });
    else if (m[2] !== undefined) out.push({ t: "strong", v: m[2] });
    else if (m[3] !== undefined) out.push({ t: "link", v: m[3], href: m[4] });
    else if (m[5] !== undefined) out.push({ t: "cite", n: m[5].split(",").map((x) => Number(x.trim())) });
    last = i + m[0].length;
  }
  if (last < s.length) out.push({ t: "text", v: s.slice(last) });
  return out;
}

export function parseMarkdown(src: string): Block[] {
  const blocks: Block[] = [];
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*```(\w*)/);
    if (fence) {
      const code: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
      i++; // closing fence (or end of a still-streaming answer)
      blocks.push({ t: "pre", lang: fence[1], code: code.join("\n") });
      continue;
    }
    const bullet = /^\s*[-*]\s+/;
    const numbered = /^\s*\d+[.)]\s+/;
    if (bullet.test(line) || numbered.test(line)) {
      const re = bullet.test(line) ? bullet : numbered;
      const items: Inline[][] = [];
      while (i < lines.length && re.test(lines[i])) items.push(parseInline(lines[i++].replace(re, "")));
      blocks.push({ t: re === bullet ? "ul" : "ol", items });
      continue;
    }
    if (!line.trim()) {
      i++;
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^\s*```/.test(lines[i]) && !bullet.test(lines[i]) && !numbered.test(lines[i]))
      para.push(lines[i++].replace(/^#+\s+/, ""));
    blocks.push({ t: "p", inl: parseInline(para.join(" ")) });
  }
  return blocks;
}

/** Markdown to readable plain text, for previews and titles. */
export function toPlain(s: string): string {
  return s
    .replace(/^\s*```.*\n?/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/^#+\s+/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
