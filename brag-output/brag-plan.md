# NexusRAG launch video — plan

**What it is:** an open-source engine that answers questions about technical documentation with cited sources, using hybrid keyword + vector search.
**For:** developers and teams with docs; on LinkedIn, the audience is recruiters and engineers.
**What sets it apart:** it was measured. The textbook hybrid recipe (equal-weight RRF) lost to plain keyword search on 5,173 held-out questions; tuned fusion won.
**Most impressive claim:** MRR@10 0.899 vs 0.873 keyword (+2.6, 95% CI +2.2 to +3.1); top-10 misses 3.5% → 1.5%.
**Visual hook:** the retrieval trace, blue (keyword) and violet (vector) dots showing where each source ranked.
**Real UI shown:** ask a question → sources appear with rank traces → cited answer streams in → click a citation.
**Tone:** polished, with one dry twist ("It lost.").
**Share caption:** The textbook hybrid search recipe lost to plain keyword search on 5,173 held-out questions. So I tuned it.

Format: 1920×1080, 30 fps, 22 s. Colours, fonts and UI markup come from `frontend/app/globals.css` and the real components. Search result shown is a real `/search` response from the app.

| # | Time | Scene | On screen |
|---|---|---|---|
| 1 | 0.0–3.0 | Hook | "Every RAG tutorial says:" / "Fuse keyword and vector search equally." Blue and violet bars slide together. |
| 2 | 3.0–6.8 | Twist | "We tested it on 5,173 held-out questions." MRR@10 bars: keyword 0.873, equal-weight hybrid 0.868. "It lost." |
| 3 | 6.8–9.8 | Reveal | Brand mark + "NexusRAG" / "Answers from your docs, with citations." |
| 4 | 9.8–15.8 | Product | Real UI: question typed, three sources slide in with rank traces animating, answer streams with citation chips, cursor clicks [1] and the source highlights. Callout: "Where each source ranked: keyword, vector, combined." |
| 5 | 15.8–19.6 | Payoff | Bars: keyword 0.873, equal-weight 0.868, tuned fusion 0.899 rises. "Tuned on held-out data: +2.6 over keyword search." "Top-10 misses: 3.5% → 1.5%" |
| 6 | 19.6–22.0 | Outro | NexusRAG + github.com/Ayaanahm/NexusRAG + stack line. |

Transitions dip through the paper background (old out, then new in). Sound: soft synth pad in A minor at 96 BPM, a muted pulse, whooshes on scene changes, quiet key ticks while typing, a chime in key when the tuned bar lands.
