import { useState } from "react";
import type { Message, Step } from "../types";

const ICON: Record<string, string> = {
  verify_customer: "🪪",
  search_policies: "🔎",
  list_invoices: "🧾",
  issue_refund: "💳",
  update_account_field: "⚙️",
  create_escalation_ticket: "🙋",
};

function StepChip({ step }: { step: Step }) {
  const cls = `step step-${step.status}`;
  return (
    <li className={cls}>
      <span className="step-icon" aria-hidden>{ICON[step.name] ?? "🔧"}</span>
      <span className="step-label">
        {step.label}
        {step.status === "running" || step.status === "retrying" ? "…" : ""}
      </span>
      {step.note && step.status === "retrying" && <span className="step-note">{step.note}</span>}
      {step.summary && step.status !== "running" && <span className="step-summary">{step.summary}</span>}
      {step.durationMs != null && step.status !== "running" && (
        <span className="step-time">{Math.round(step.durationMs)} ms{step.attempts && step.attempts > 1 ? ` · ${step.attempts} tries` : ""}</span>
      )}
      <span className="step-state" aria-label={step.status}>
        {step.status === "done" ? "✓" : step.status === "failed" ? "✕" : <span className="spinner" />}
      </span>
    </li>
  );
}

function renderText(text: string, onCite: (id: string) => void) {
  const parts = text.split(/(\[[a-z0-9-]+:\d+\])/g);
  return parts.map((p, i) => {
    const m = p.match(/^\[([a-z0-9-]+:\d+)\]$/);
    return m ? (
      <button key={i} className="cite" onClick={() => onCite(m[1])} title="Show source">{m[1]}</button>
    ) : (
      <span key={i}>{p}</span>
    );
  });
}

export function MessageView({ m, onRetry }: { m: Message; onRetry: (text: string) => void }) {
  const [openCite, setOpenCite] = useState<string | null>(null);
  if (m.role === "user") {
    return (
      <div className="row row-user">
        <div className="bubble bubble-user">
          {m.mode === "voice" && <span className="mode-tag" title="Spoken">🎙</span>}
          {m.text || <span className="muted">Listening…</span>}
        </div>
      </div>
    );
  }
  const g = m.grounding;
  const cite = m.citations.find((c) => c.id === openCite);
  return (
    <div className="row row-agent">
      <div className="avatar" aria-hidden>O</div>
      <div className="agent-col">
        {m.steps.length > 0 && <ul className="steps">{m.steps.map((s) => <StepChip key={s.id} step={s} />)}</ul>}
        {(m.text || m.streaming) && (
          <div className={`bubble bubble-agent${m.replaced ? " replaced" : ""}`}>
            {m.mode === "voice" && <span className="mode-tag" title="Spoken">🔊</span>}
            {m.text ? renderText(m.text, (id) => setOpenCite(openCite === id ? null : id)) : null}
            {m.streaming && !m.text && <span className="typing"><i /><i /><i /></span>}
            {m.interrupted && <span className="interrupted">interrupted</span>}
          </div>
        )}
        {m.streaming && m.status && <div className="live-status"><span className="spinner" /> {m.status}</div>}
        {cite && (
          <div className="source-card">
            <div className="source-head"><b>{cite.id}</b> · {cite.source} <span className="score">similarity {cite.score.toFixed(2)}</span></div>
            <p>{cite.text}</p>
          </div>
        )}
        {m.escalation && (
          <div className="ticket">
            <div className="ticket-head">🙋 Escalated to a human · <b>{m.escalation.ticket_id}</b></div>
            <div className="ticket-body">
              <span className={`pill pri-${m.escalation.priority}`}>{m.escalation.priority}</span>
              <span className="pill">{m.escalation.category}</span>
              <span className="pill">first reply within {m.escalation.first_response_sla}</span>
            </div>
            <div className="ticket-summary">{m.escalation.summary}</div>
          </div>
        )}
        {g && g.status !== "checking" && !m.streaming && (
          <div className={`grounding g-${g.status}`}>
            {g.status === "grounded" && <>✓ Grounded{m.citations.length ? ` in ${m.citations.length} source${m.citations.length > 1 ? "s" : ""}` : ""}</>}
            {g.status === "ungrounded" && <>⚠ Draft failed the grounding check and was replaced: {g.reason}</>}
            {g.status === "unverified" && <>◌ Not verified: {g.reason}</>}
            {g.confidence && <span className={`conf conf-${g.confidence}`}>retrieval: {g.confidence}</span>}
          </div>
        )}
        {m.error && (
          <div className="error-card">
            <span>⚠ {m.error.message}</span>
            {m.error.retryable && m.error.retryText && <button className="btn-small" onClick={() => onRetry(m.error!.retryText!)}>Retry</button>}
          </div>
        )}
      </div>
    </div>
  );
}
