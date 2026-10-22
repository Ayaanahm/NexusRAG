// The retrieval trace: where a passage ranked for keyword search, for vector
// search, and after fusion, on one shared 1..depth scale. A dot on the left
// means "ranked near the top"; a hollow end-cap means "not in that list".

type Props = { bm25: number | null; dense: number | null; fused: number | null; depth?: number };

const ROWS = [
  { key: "bm25", label: "Keyword", cls: "kw" },
  { key: "dense", label: "Vector", cls: "vec" },
  { key: "fused", label: "Combined", cls: "fused" },
] as const;

export function RankTrace({ bm25, dense, fused, depth = 50 }: Props) {
  const values = { bm25, dense, fused };
  const summary = ROWS.map((r) => `${r.label} ${values[r.key] ? `#${values[r.key]}` : "not ranked"}`).join(", ");
  return (
    <div className="trace" role="img" aria-label={`Ranks: ${summary}`}>
      {ROWS.map((r) => {
        const v = values[r.key];
        const pos = v ? Math.min(1, (Math.log(v) / Math.log(depth))) : 1;
        return (
          <div className="trace-row" key={r.key}>
            <span className="trace-label">{r.label}</span>
            <span className="trace-track">
              <span className={`trace-dot ${r.cls} ${v ? "" : "missing"}`} style={{ left: `${pos * 100}%` }} />
            </span>
            <span className="trace-value">{v ? `#${v}` : "–"}</span>
          </div>
        );
      })}
    </div>
  );
}
