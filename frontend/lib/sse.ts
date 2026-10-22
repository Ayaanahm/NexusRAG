// Minimal Server-Sent Events parser for fetch() streams. EventSource only
// supports GET, and /chat is a POST with a JSON body.

export type SSEEvent = { event: string; data: string };

export class SSEParser {
  private buffer = "";

  /** Feed a decoded text chunk; returns every event completed by it. */
  push(chunk: string): SSEEvent[] {
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    const events: SSEEvent[] = [];
    let sep: number;
    while ((sep = this.buffer.indexOf("\n\n")) !== -1) {
      const block = this.buffer.slice(0, sep);
      this.buffer = this.buffer.slice(sep + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith(":")) continue; // comment / keep-alive ping
        const colon = line.indexOf(":");
        const field = colon === -1 ? line : line.slice(0, colon);
        let value = colon === -1 ? "" : line.slice(colon + 1);
        if (value.startsWith(" ")) value = value.slice(1);
        if (field === "event") event = value;
        else if (field === "data") data.push(value);
      }
      if (data.length) events.push({ event, data: data.join("\n") });
    }
    return events;
  }
}

export async function* readSSE(response: Response): AsyncGenerator<SSEEvent> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const ev of parser.push(decoder.decode(value, { stream: true }))) yield ev;
  }
}
