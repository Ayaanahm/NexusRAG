import { describe, expect, it } from "vitest";
import { parseInline, parseMarkdown, toPlain } from "./markdown";
import { SSEParser } from "./sse";

describe("SSEParser", () => {
  it("handles events split across chunks, CRLF and pings", () => {
    const p = new SSEParser();
    expect(p.push("event: tok")).toEqual([]);
    expect(p.push('en\r\ndata: {"text":"a"}\r\n\r\n: ping\n\nevent: done\ndata: {}\n\n')).toEqual([
      { event: "token", data: '{"text":"a"}' },
      { event: "done", data: "{}" },
    ]);
  });

  it("joins multi-line data", () => {
    expect(new SSEParser().push("data: a\ndata: b\n\n")).toEqual([{ event: "message", data: "a\nb" }]);
  });
});

describe("markdown", () => {
  it("parses citations, code and links without HTML", () => {
    expect(parseInline("Use `X` [1, 3] see [docs](https://a.dev) <b>")).toEqual([
      { t: "text", v: "Use " },
      { t: "code", v: "X" },
      { t: "text", v: " " },
      { t: "cite", n: [1, 3] },
      { t: "text", v: " see " },
      { t: "link", v: "docs", href: "https://a.dev" },
      { t: "text", v: " <b>" },
    ]);
  });

  it("does not turn javascript: links into links", () => {
    expect(parseInline("[x](javascript:alert(1))").some((i) => i.t === "link")).toBe(false);
  });

  it("parses blocks including an unterminated fence while streaming", () => {
    const b = parseMarkdown("Intro [1].\n\n- one\n- two\n\n```python\nx = 1");
    expect(b.map((x) => x.t)).toEqual(["p", "ul", "pre"]);
    expect(b[2]).toEqual({ t: "pre", lang: "python", code: "x = 1" });
  });
});

describe("toPlain", () => {
  it("strips markdown syntax", () => {
    expect(toPlain("**FastAPI**'s `HTTPException` [docs](https://a.dev)\n```python\nx = 1\n```")).toBe(
      "FastAPI's HTTPException docs\nx = 1",
    );
  });
});
