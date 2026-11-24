import type { AppConfig, Simulation } from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string, public retryAfter?: number) {
    super(message);
  }
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, body.message ?? `Request failed (${res.status})`, body.retry_after);
  return body as T;
}

export const getConfig = () => json<AppConfig>("/api/config");
export const createSession = () => json<{ session_id: string }>("/api/sessions", { method: "POST" });

export const voiceSession = (sessionId: string) =>
  json<{ client_secret: string; expires_at: number; model: string; max_session_s: number }>("/api/voice/session", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId }),
  });

export const voiceTool = (sessionId: string, callId: string, name: string, args: string) =>
  json<{ output: string; ok: boolean; label: string; summary: string; duration_ms: number; retrieval: unknown; escalation: unknown }>(
    "/api/voice/tool",
    { method: "POST", body: JSON.stringify({ session_id: sessionId, call_id: callId, name, arguments: args }) },
  );

export const voiceTranscript = (sessionId: string, role: "user" | "assistant", text: string, interrupted = false) =>
  json("/api/voice/transcript", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, role, text, interrupted }),
  }).catch(() => undefined);

export const voiceUsage = (sessionId: string, payload: Record<string, unknown>) =>
  json("/api/voice/usage", { method: "POST", body: JSON.stringify({ session_id: sessionId, ...payload }) });

export type SseHandler = (event: string, data: any) => void;

/** POST + Server-Sent Events. EventSource can't POST, so parse the stream by hand. */
export async function streamChat(
  sessionId: string,
  message: string,
  simulate: Simulation,
  onEvent: SseHandler,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message, simulate }),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
  }
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.message ?? `Request failed (${res.status})`, body.retry_after);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trim());
      }
      if (data.length) onEvent(event, JSON.parse(data.join("\n")));
    }
  }
}
