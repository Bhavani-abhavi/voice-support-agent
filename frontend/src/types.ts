export type Mode = "chat" | "voice";
export type Simulation = "none" | "tool_flaky" | "tool_down" | "llm_rate_limit" | "llm_timeout";

export interface Retrieval {
  query: string;
  tier_filter: string[];
  confidence: "high" | "borderline" | "low";
  top_score: number;
  thresholds: [number, number];
  scores: number[];
  chunks: string[];
  latency_ms: number;
}

export interface Step {
  id: string;
  name: string;
  label: string;
  status: "running" | "retrying" | "done" | "failed";
  durationMs?: number;
  summary?: string;
  attempts?: number;
  note?: string;
  retrieval?: Retrieval | null;
}

export interface Citation {
  id: string;
  source: string;
  text: string;
  score: number;
}

export interface Ticket {
  ticket_id: string;
  priority: string;
  category: string;
  summary: string;
  first_response_sla: string;
  created_via?: string;
}

export interface Grounding {
  status: "checking" | "grounded" | "ungrounded" | "unverified";
  method?: string;
  reason?: string;
  confidence?: "high" | "borderline" | "low" | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  mode: Mode;
  text: string;
  streaming: boolean;
  steps: Step[];
  status?: string; // live state label while the agent works
  citations: Citation[];
  grounding?: Grounding;
  escalation?: Ticket;
  error?: { message: string; retryable: boolean; retryText?: string };
  interrupted?: boolean;
  replaced?: boolean;
}

export interface LlmCall {
  purpose: string;
  model: string;
  ttft_ms: number | null;
  duration_ms: number | null;
  input_tokens: number;
  output_tokens: number;
  cached_tokens: number;
  cost_usd: number;
  priced: boolean;
  prompt: Array<Record<string, unknown>>;
  output: unknown;
}

export interface Span {
  name: string;
  kind: "llm" | "tool" | "verify" | string;
  start_ms?: number;
  duration_ms: number;
  status: string;
  summary?: string;
}

export interface Trace {
  id: string;
  mode: Mode;
  user_text: string;
  outcome: string;
  latency_ms: number | null;
  ttft_ms: number | null;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  llm_calls: LlmCall[];
  spans: Span[];
  retrievals: Retrieval[];
  guardrail: Record<string, unknown>;
  errors: string[];
  token_detail?: Record<string, number>;
}

export interface DemoAccount {
  name: string;
  email: string;
  zip: string;
  tier: string;
  scenario: string;
}

export interface AppConfig {
  provider: "openai" | "mock";
  chat_model: string;
  voice_enabled: boolean;
  realtime_model: string | null;
  vector_store: string;
  embedding_model: string;
  thresholds: { escalate_below: number; answer_at: number };
  demo_accounts: DemoAccount[];
  sample_prompts: { label: string; text: string }[];
  voice_max_session_s: number;
}

export type VoiceState = "off" | "connecting" | "listening" | "user_speaking" | "thinking" | "speaking" | "error";
