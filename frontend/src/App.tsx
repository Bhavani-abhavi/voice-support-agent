import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { ApiError, createSession, getConfig, streamChat } from "./api";
import { Drawer } from "./components/Drawer";
import { MessageView } from "./components/MessageView";
import { Waveform } from "./components/Waveform";
import { RealtimeClient } from "./realtime";
import { initialState, reducer, uid } from "./store";
import type { AppConfig, Mode, Simulation, VoiceState } from "./types";

const SIMULATIONS: { value: Simulation; label: string; hint: string }[] = [
  { value: "none", label: "None", hint: "Normal operation" },
  { value: "tool_flaky", label: "Flaky tool", hint: "First call to each tool times out, then the retry succeeds" },
  { value: "tool_down", label: "Billing system down", hint: "Invoice/refund tools fail every attempt; the agent must not invent results" },
  { value: "llm_rate_limit", label: "LLM rate limit", hint: "First model call gets a 429; backoff and retry" },
  { value: "llm_timeout", label: "LLM timeout", hint: "Every model call times out; graceful error with retry" },
];

const VOICE_LABEL: Record<VoiceState, string> = {
  off: "Voice is off",
  connecting: "Connecting…",
  listening: "Listening. Go ahead and talk.",
  user_speaking: "Hearing you…",
  thinking: "Thinking…",
  speaking: "Speaking. Talk anytime to interrupt.",
  error: "Voice error",
};

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [sessionId, setSessionId] = useState<string>("");
  const [bootError, setBootError] = useState<string>("");
  const [state, dispatch] = useReducer(reducer, initialState);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("chat");
  const [simulate, setSimulate] = useState<Simulation>("none");
  const [drawer, setDrawer] = useState(() => window.innerWidth > 1100);
  const [sidebar, setSidebar] = useState(() => window.innerWidth > 900);
  const [voiceState, setVoiceState] = useState<VoiceState>("off");
  const [voiceNote, setVoiceNote] = useState("");
  const [, force] = useState(0);
  const voiceRef = useRef<RealtimeClient | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const voiceMsg = useRef(new Map<string, string>()); // realtime response id -> message id
  const toolMsg = useRef(new Map<string, string>()); // call id -> message id
  const lastToolMsg = useRef<string | null>(null);

  const boot = useCallback(async () => {
    setBootError("");
    try {
      const [cfg, s] = await Promise.all([getConfig(), createSession()]);
      setConfig(cfg);
      setSessionId(s.session_id);
    } catch (e) {
      setBootError((e as Error).message);
    }
  }, []);
  useEffect(() => { boot(); }, [boot]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [state.messages]);

  // ---------------------------------------------------------------- chat
  const send = useCallback(async (text: string) => {
    const msg = text.trim();
    if (!msg || !sessionId || state.busy) return;
    setInput("");
    const aid = uid("a");
    dispatch({ type: "user", id: uid("u"), text: msg, mode: "chat" });
    dispatch({ type: "assistant_start", id: aid, mode: "chat" });
    dispatch({ type: "busy", busy: true });
    try {
      await streamChat(sessionId, msg, simulate, (event, data) => {
        switch (event) {
          case "status": dispatch({ type: "status", id: aid, label: data.label }); break;
          case "token": dispatch({ type: "token", id: aid, text: data.text }); break;
          case "tool_start":
            dispatch({ type: "step_start", id: aid, step: { id: data.call_id, name: data.name, label: data.label, status: "running" } });
            break;
          case "tool_retry":
            dispatch({ type: "step_update", id: aid, stepId: data.call_id, patch: { status: "retrying", note: data.message } });
            break;
          case "tool_end":
            dispatch({
              type: "step_update", id: aid, stepId: data.call_id,
              patch: { status: data.ok ? "done" : "failed", durationMs: data.duration_ms, summary: data.summary, attempts: data.attempts, retrieval: data.retrieval },
            });
            break;
          case "final": dispatch({ type: "set_text", id: aid, text: data.text, replaced: data.replaced }); break;
          case "escalation": dispatch({ type: "escalation", id: aid, ticket: data }); break;
          case "verification":
            dispatch({ type: "verification", id: aid, grounding: { status: data.status, method: data.method, reason: data.reason, confidence: data.confidence }, citations: data.citations });
            break;
          case "error": dispatch({ type: "error", id: aid, message: data.message, retryable: data.retryable, retryText: msg }); break;
          case "trace": dispatch({ type: "trace", trace: data }); break;
          case "done": dispatch({ type: "assistant_done", id: aid }); break;
        }
      });
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 404) {
        dispatch({ type: "error", id: aid, message: "Your session expired, so I started a new one. Please send that again.", retryable: true, retryText: msg });
        const s = await createSession().catch(() => null);
        if (s) setSessionId(s.session_id);
      } else {
        dispatch({ type: "error", id: aid, message: err.message, retryable: err.status !== 413, retryText: msg });
      }
    } finally {
      dispatch({ type: "assistant_done", id: aid });
      dispatch({ type: "busy", busy: false });
    }
  }, [sessionId, simulate, state.busy]);

  // ---------------------------------------------------------------- voice
  const startVoice = useCallback(async () => {
    if (!sessionId || voiceRef.current) return;
    setVoiceNote("");
    const client = new RealtimeClient({
      onState: (s) => { setVoiceState(s); force((n) => n + 1); },
      onUserSpeechStart: (itemId) => dispatch({ type: "user", id: `vu-${itemId}`, text: "", mode: "voice" }),
      onUserTranscript: (itemId, text) => dispatch({ type: "user", id: `vu-${itemId}`, text, mode: "voice" }),
      onAssistantStart: (responseId) => {
        // A response that only called tools hands over to the follow-up response: keep one bubble.
        const reuse = lastToolMsg.current;
        const id = reuse ?? `va-${responseId}`;
        voiceMsg.current.set(responseId, id);
        if (!reuse) dispatch({ type: "assistant_start", id, mode: "voice" });
        else dispatch({ type: "status", id, label: "Answering" });
      },
      onAssistantDelta: (responseId, delta) => {
        const id = voiceMsg.current.get(responseId);
        if (id) { lastToolMsg.current = null; dispatch({ type: "token", id, text: delta }); }
      },
      onAssistantDone: (responseId, _text, interrupted) => {
        const id = voiceMsg.current.get(responseId);
        if (!id) return;
        const awaitingTools = [...toolMsg.current.values()].includes(id) && lastToolMsg.current === id;
        if (!awaitingTools) {
          dispatch({ type: "assistant_done", id, interrupted });
          dispatch({ type: "drop_if_empty", id });
        }
      },
      onToolStart: (responseId, callId, name) => {
        const id = voiceMsg.current.get(responseId);
        if (!id) return;
        toolMsg.current.set(callId, id);
        lastToolMsg.current = id;
        dispatch({ type: "step_start", id, step: { id: callId, name, label: name.replace(/_/g, " "), status: "running" } });
      },
      onToolEnd: (callId, result, error) => {
        const id = toolMsg.current.get(callId);
        if (!id) return;
        dispatch({
          type: "step_update", id, stepId: callId,
          patch: result
            ? { status: result.ok ? "done" : "failed", label: result.label, summary: result.summary, durationMs: result.duration_ms }
            : { status: "failed", summary: error },
        });
        if (result?.escalation) dispatch({ type: "escalation", id, ticket: result.escalation as any });
      },
      onTrace: (trace) => dispatch({ type: "trace", trace }),
      onError: (message) => setVoiceNote(message),
      onEnded: (reason) => {
        voiceRef.current = null;
        if (reason !== "ended" && reason !== "error") setVoiceNote(reason);
        lastToolMsg.current = null;
      },
    });
    voiceRef.current = client;
    try {
      await client.connect(sessionId);
    } catch (e) {
      const err = e as Error;
      client.disconnect("error");
      voiceRef.current = null;
      setVoiceState("off");
      setVoiceNote(err.name === "NotAllowedError" ? "Microphone access was blocked. Allow it in your browser's address bar and try again." : err.message);
    }
  }, [sessionId]);

  const stopVoice = useCallback(() => { voiceRef.current?.disconnect("ended"); }, []);

  const switchMode = (m: Mode) => {
    if (m === mode) return;
    if (m === "chat") stopVoice();
    setMode(m);
  };
  useEffect(() => () => voiceRef.current?.disconnect("ended"), []);

  const voiceOn = voiceState !== "off" && voiceState !== "error";
  const provider = config?.provider;

  if (bootError) {
    return (
      <div className="boot">
        <h1>Orbit Support</h1>
        <p>{bootError}</p>
        <button className="btn" onClick={boot}>Try again</button>
      </div>
    );
  }

  return (
    <div className={`app${drawer ? " with-drawer" : ""}${sidebar ? " with-sidebar" : ""}`}>
      <header className="topbar">
        <button className="icon-btn hide-wide" onClick={() => setSidebar(!sidebar)} aria-label="Toggle demo panel">☰</button>
        <div className="brand"><span className="logo" aria-hidden /> Orbit Support <span className="sub">AI agent console</span></div>
        <div className="seg" role="tablist" aria-label="Interaction mode">
          <button role="tab" aria-selected={mode === "chat"} className={mode === "chat" ? "on" : ""} onClick={() => switchMode("chat")}>💬 Chat</button>
          <button role="tab" aria-selected={mode === "voice"} className={mode === "voice" ? "on" : ""} onClick={() => switchMode("voice")}>🎙 Voice</button>
        </div>
        <div className="top-right">
          {provider && <span className={`prov prov-${provider}`} title={provider === "mock" ? "No API key: scripted offline agent" : config?.chat_model}>{provider === "mock" ? "offline mock" : "OpenAI"}</span>}
          <button className={`btn-ghost${drawer ? " on" : ""}`} onClick={() => setDrawer(!drawer)} aria-expanded={drawer}>📈 Observability</button>
        </div>
      </header>

      {sidebar && (
        <aside className="sidebar" aria-label="Demo controls">
          <section>
            <h3>Demo accounts</h3>
            <p className="muted small">Mock CRM. Verify with email + ZIP.</p>
            {config?.demo_accounts.map((a) => (
              <button key={a.email} className="acct" onClick={() => setInput(`I'm ${a.email}, ZIP ${a.zip}. `)}>
                <b>{a.name}</b> <span className="tier">{a.tier}</span>
                <span className="small muted">{a.scenario}</span>
              </button>
            ))}
          </section>
          <section>
            <h3>Simulate failure</h3>
            <select value={simulate} onChange={(e) => setSimulate(e.target.value as Simulation)} aria-label="Failure simulation">
              {SIMULATIONS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
            <p className="muted small">{SIMULATIONS.find((s) => s.value === simulate)?.hint}. Applies to chat turns.</p>
          </section>
          <section className="how">
            <h3>How it works</h3>
            <ol className="small">
              <li><b>Agent loop</b>: the model chains tools (verify → search → invoices → refund); limits are enforced in code.</li>
              <li><b>Retrieval gate</b>: Qdrant similarity below threshold → passages withheld → human handoff.</li>
              <li><b>Grounding check</b>: each answer is verified against retrieved sources before it's final.</li>
              <li><b>One session</b>: chat and voice share memory and tools.</li>
            </ol>
          </section>
        </aside>
      )}

      <main className="main">
        <div className="messages" ref={listRef}>
          {state.messages.length === 0 && (
            <div className="empty">
              <h1>How can we help?</h1>
              <p className="muted">Ask about plans, billing or sync, or try a scenario. Switch to voice anytime; the conversation carries over.</p>
              <div className="chips">
                {config?.sample_prompts.map((p) => (
                  <button key={p.label} className="chip" onClick={() => (mode === "chat" ? send(p.text) : setInput(p.text))}>{p.label}</button>
                ))}
              </div>
            </div>
          )}
          {state.messages.map((m) => <MessageView key={m.id} m={m} onRetry={send} />)}
        </div>

        {mode === "chat" ? (
          <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input); }}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }}
              placeholder={state.busy ? "Orbit is working…" : "Message Orbit support…"}
              rows={1}
              maxLength={2000}
              aria-label="Message"
            />
            <button className="btn" type="submit" disabled={!input.trim() || state.busy || !sessionId}>Send</button>
          </form>
        ) : (
          <div className={`voice-dock vs-${voiceState}`}>
            {!config?.voice_enabled ? (
              <p className="muted">Voice mode needs an OpenAI key on the server (it's running in offline mock mode). Chat works fully.</p>
            ) : (
              <>
                <button
                  className={`mic${voiceOn ? " live" : ""}`}
                  onClick={voiceOn ? stopVoice : startVoice}
                  disabled={voiceState === "connecting"}
                  aria-label={voiceOn ? "End voice session" : "Start voice session"}
                >
                  {voiceOn ? "■" : "🎙"}
                </button>
                <div className="voice-main">
                  <div className="voice-label">{voiceState === "off" ? "Tap the mic to talk to Orbit" : VOICE_LABEL[voiceState]}</div>
                  <Waveform mic={voiceRef.current?.micAnalyser} out={voiceRef.current?.outAnalyser} active={voiceOn} />
                  <div className="legend small"><span className="lg-user" /> you <span className="lg-agent" /> Orbit {voiceNote && <span className="voice-note">· {voiceNote}</span>}</div>
                </div>
              </>
            )}
          </div>
        )}
      </main>

      {drawer && <Drawer traces={state.traces} config={config} onClose={() => setDrawer(false)} />}
    </div>
  );
}
