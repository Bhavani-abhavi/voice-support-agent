import { useState } from "react";
import type { AppConfig, Trace } from "../types";

const fmtMs = (v: number | null | undefined) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);
const fmtUsd = (v: number) => (v === 0 ? "$0" : v < 0.01 ? `$${v.toFixed(5)}` : `$${v.toFixed(4)}`);

function TraceCard({ t, index }: { t: Trace; index: number }) {
  const [open, setOpen] = useState(false);
  const [prompt, setPrompt] = useState<number | null>(null);
  const total = Math.max(t.latency_ms ?? 0, ...t.spans.map((s) => (s.start_ms ?? 0) + s.duration_ms), 1);
  const g = t.guardrail as Record<string, any>;
  return (
    <div className="trace">
      <button className="trace-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={`badge badge-${t.mode}`}>{t.mode === "voice" ? "🎙 voice" : "💬 chat"}</span>
        <span className="trace-q">#{index + 1} {t.user_text || "(voice turn)"}</span>
        <span className={`outcome o-${t.outcome}`}>{t.outcome}</span>
      </button>
      <div className="metrics">
        <div><label>{t.mode === "voice" ? "speech end → audio" : "first token"}</label><b>{fmtMs(t.ttft_ms)}</b></div>
        <div><label>end-to-end</label><b>{fmtMs(t.latency_ms)}</b></div>
        <div><label>tokens in / out</label><b>{t.input_tokens.toLocaleString()} / {t.output_tokens.toLocaleString()}</b></div>
        <div><label>est. cost</label><b>{fmtUsd(t.cost_usd)}</b></div>
      </div>
      {t.token_detail && (
        <div className="token-detail">audio in {t.token_detail.audio_in} · text in {t.token_detail.text_in} · cached {t.token_detail.cached_in} · audio out {t.token_detail.audio_out} · text out {t.token_detail.text_out}</div>
      )}
      {t.spans.length > 0 && (
        <div className="waterfall">
          {t.spans.map((s, i) => (
            <div className="wf-row" key={i} title={`${s.name} · ${fmtMs(s.duration_ms)}${s.summary ? " · " + s.summary : ""}`}>
              <span className="wf-name">{s.name}</span>
              <span className="wf-track">
                <span
                  className={`wf-bar k-${s.kind} s-${s.status}`}
                  style={{ left: `${((s.start_ms ?? 0) / total) * 100}%`, width: `${Math.max((s.duration_ms / total) * 100, 1.5)}%` }}
                />
              </span>
              <span className="wf-ms">{fmtMs(s.duration_ms)}</span>
            </div>
          ))}
        </div>
      )}
      {g && Object.keys(g).length > 0 && (
        <div className="guard">
          <b>Guardrails</b>
          {g.retrieval_confidence !== undefined && <span>retrieval: <i className={`conf conf-${g.retrieval_confidence}`}>{g.retrieval_confidence ?? "n/a"}</i></span>}
          {g.verification && <span>verification: <i>{g.verification}</i>{g.method ? ` (${g.method})` : ""}</span>}
          {g.escalated && <span>→ escalated to a human</span>}
          {g.reason && <div className="muted small">{g.reason}</div>}
        </div>
      )}
      {open && (
        <div className="trace-detail">
          {t.retrievals.map((r, i) => (
            <div className="retrieval" key={i}>
              <div><b>Retrieval</b> “{r.query}”</div>
              <div className="small">filter: tiers ∈ [{r.tier_filter.join(", ")}] · {fmtMs(r.latency_ms)}</div>
              <div className="score-bar">
                <span className="zone zone-low" style={{ width: `${r.thresholds[0] * 100}%` }} />
                <span className="zone zone-mid" style={{ width: `${(r.thresholds[1] - r.thresholds[0]) * 100}%` }} />
                <span className="zone zone-high" />
                {r.scores.map((s, j) => <span key={j} className={`dot${j === 0 ? " top" : ""}`} style={{ left: `${Math.min(s, 1) * 100}%` }} title={s.toFixed(3)} />)}
              </div>
              <div className="small">top {r.top_score.toFixed(3)} → <b>{r.confidence}</b> (escalate &lt; {r.thresholds[0]}, answer ≥ {r.thresholds[1]}) · chunks: {r.chunks.join(", ") || "withheld"}</div>
            </div>
          ))}
          {t.llm_calls.map((c, i) => (
            <div className="llm-call" key={i}>
              <div className="llm-head">
                <b>{c.purpose}</b> · {c.model} · {fmtMs(c.duration_ms)} · {c.input_tokens}→{c.output_tokens} tok · {fmtUsd(c.cost_usd)}{!c.priced && " (unpriced)"}
                <button className="btn-link" onClick={() => setPrompt(prompt === i ? null : i)}>{prompt === i ? "hide prompt" : "view prompt"}</button>
              </div>
              {prompt === i && <pre className="prompt">{JSON.stringify({ messages: c.prompt, output: c.output }, null, 2)}</pre>}
            </div>
          ))}
          {t.errors.length > 0 && <div className="small err">errors: {t.errors.join(", ")}</div>}
        </div>
      )}
    </div>
  );
}

export function Drawer({ traces, config, onClose }: { traces: Trace[]; config: AppConfig | null; onClose: () => void }) {
  const cost = traces.reduce((a, t) => a + t.cost_usd, 0);
  const tokens = traces.reduce((a, t) => a + t.input_tokens + t.output_tokens, 0);
  const lat = traces.filter((t) => t.latency_ms != null);
  const avg = lat.length ? lat.reduce((a, t) => a + (t.latency_ms ?? 0), 0) / lat.length : null;
  return (
    <aside className="drawer" aria-label="Observability">
      <div className="drawer-head">
        <h2>Observability</h2>
        <button className="icon-btn" onClick={onClose} aria-label="Close observability panel">✕</button>
      </div>
      <div className="totals">
        <div><label>turns</label><b>{traces.length}</b></div>
        <div><label>avg latency</label><b>{fmtMs(avg)}</b></div>
        <div><label>tokens</label><b>{tokens.toLocaleString()}</b></div>
        <div><label>session cost</label><b>{fmtUsd(cost)}</b></div>
      </div>
      {config && (
        <div className="stack small muted">
          {config.chat_model} · {config.embedding_model} · {config.vector_store}
          {config.realtime_model ? ` · ${config.realtime_model}` : ""}
        </div>
      )}
      <div className="trace-list">
        {traces.length === 0 && <p className="muted small">Send a message. Each turn's prompt trace, latency waterfall, token usage and cost appear here.</p>}
        {[...traces].reverse().map((t) => <TraceCard key={t.id} t={t} index={traces.indexOf(t)} />)}
      </div>
    </aside>
  );
}
