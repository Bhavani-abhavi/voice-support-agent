import type { Citation, Grounding, Message, Mode, Step, Ticket, Trace } from "./types";

export interface State {
  messages: Message[];
  traces: Trace[];
  busy: boolean;
}

export const initialState: State = { messages: [], traces: [], busy: false };

let seq = 0;
export const uid = (p: string) => `${p}-${Date.now().toString(36)}-${(seq++).toString(36)}`;

export type Action =
  | { type: "user"; id: string; text: string; mode: Mode }
  | { type: "assistant_start"; id: string; mode: Mode }
  | { type: "status"; id: string; label: string }
  | { type: "token"; id: string; text: string }
  | { type: "set_text"; id: string; text: string; replaced?: boolean }
  | { type: "step_start"; id: string; step: Step }
  | { type: "step_update"; id: string; stepId: string; patch: Partial<Step> }
  | { type: "verification"; id: string; grounding: Grounding; citations: Citation[] }
  | { type: "escalation"; id: string; ticket: Ticket }
  | { type: "error"; id: string; message: string; retryable: boolean; retryText?: string }
  | { type: "assistant_done"; id: string; interrupted?: boolean }
  | { type: "drop_if_empty"; id: string }
  | { type: "trace"; trace: Trace }
  | { type: "busy"; busy: boolean };

const patch = (s: State, id: string, f: (m: Message) => Message): State => ({
  ...s,
  messages: s.messages.map((m) => (m.id === id ? f(m) : m)),
});

const blank = (id: string, role: Message["role"], mode: Mode, text = ""): Message => ({
  id, role, mode, text, streaming: role === "assistant", steps: [], citations: [],
});

export function reducer(s: State, a: Action): State {
  switch (a.type) {
    case "user":
      if (s.messages.some((m) => m.id === a.id)) return patch(s, a.id, (m) => ({ ...m, text: a.text }));
      return { ...s, messages: [...s.messages, blank(a.id, "user", a.mode, a.text)] };
    case "assistant_start":
      if (s.messages.some((m) => m.id === a.id)) return patch(s, a.id, (m) => ({ ...m, streaming: true }));
      return { ...s, messages: [...s.messages, blank(a.id, "assistant", a.mode)] };
    case "status":
      return patch(s, a.id, (m) => ({ ...m, status: a.label }));
    case "token":
      return patch(s, a.id, (m) => ({ ...m, text: m.text + a.text }));
    case "set_text":
      return patch(s, a.id, (m) => ({ ...m, text: a.text, replaced: a.replaced ?? m.replaced }));
    case "step_start":
      return patch(s, a.id, (m) => ({ ...m, steps: [...m.steps, a.step] }));
    case "step_update":
      return patch(s, a.id, (m) => ({
        ...m,
        steps: m.steps.map((st) => (st.id === a.stepId ? { ...st, ...a.patch } : st)),
      }));
    case "verification":
      return patch(s, a.id, (m) => ({ ...m, grounding: a.grounding, citations: a.citations }));
    case "escalation":
      return patch(s, a.id, (m) => ({ ...m, escalation: a.ticket }));
    case "error":
      return patch(s, a.id, (m) => ({
        ...m, streaming: false, status: undefined,
        error: { message: a.message, retryable: a.retryable, retryText: a.retryText },
      }));
    case "assistant_done":
      return patch(s, a.id, (m) => ({ ...m, streaming: false, status: undefined, interrupted: a.interrupted ?? m.interrupted }));
    case "drop_if_empty":
      return { ...s, messages: s.messages.filter((m) => m.id !== a.id || m.text || m.steps.length || m.error) };
    case "trace":
      return { ...s, traces: [...s.traces.filter((t) => t.id !== a.trace.id), a.trace] };
    case "busy":
      return { ...s, busy: a.busy };
  }
}
