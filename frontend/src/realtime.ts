/**
 * Live voice over the OpenAI Realtime API (WebRTC).
 *
 * Audio flows browser <-> OpenAI directly, so there's no relay hop and the lowest
 * latency. Decisions still go through our backend: the client secret is minted
 * there, and every function call is executed there with the same guardrails as chat.
 */
import { voiceSession, voiceTool, voiceTranscript, voiceUsage } from "./api";
import type { Trace, VoiceState } from "./types";

export interface RealtimeCallbacks {
  onState: (state: VoiceState) => void;
  onUserSpeechStart: (itemId: string) => void;
  onUserTranscript: (itemId: string, text: string, final: boolean) => void;
  onAssistantStart: (responseId: string) => void;
  onAssistantDelta: (responseId: string, delta: string) => void;
  onAssistantDone: (responseId: string, text: string, interrupted: boolean) => void;
  onInterrupted: (responseId: string) => void;
  onToolStart: (responseId: string, callId: string, name: string) => void;
  onToolEnd: (callId: string, result: Awaited<ReturnType<typeof voiceTool>> | null, error?: string) => void;
  onTrace: (trace: Trace) => void;
  onError: (message: string) => void;
  onEnded: (reason: string) => void;
}

type Evt = Record<string, any>;

export class RealtimeClient {
  private pc?: RTCPeerConnection;
  private dc?: RTCDataChannel;
  private mic?: MediaStream;
  private audio?: HTMLAudioElement;
  private ctx?: AudioContext;
  micAnalyser?: AnalyserNode;
  outAnalyser?: AnalyserNode;
  private sessionId = "";
  private state: VoiceState = "off";
  private speaking = false;
  private interrupted = new Set<string>();
  private pendingTools = new Map<string, Promise<void>[]>();
  private assistantText = new Map<string, string>();
  private userText = new Map<string, string>();
  private lastUserText = "";
  private tSpeechStopped = 0;
  private tFirstAudio = 0;
  private timer?: number;
  private closed = false;
  private responseActive = false;
  private lastResponseId = "";
  private audioItem: { id: string; playStart: number } | null = null;
  private muted = false;
  private interruptAt = 0;

  constructor(private cb: RealtimeCallbacks) {}

  /** Agent audio is audible: server playback events, or energy on the remote track as a fallback. */
  private agentAudible(): boolean {
    if (this.speaking) return true;
    const node = this.outAnalyser;
    if (!node) return false;
    const buf = new Uint8Array(node.fftSize);
    node.getByteTimeDomainData(buf);
    let sum = 0;
    for (const v of buf) {
      const d = (v - 128) / 128;
      sum += d * d;
    }
    return Math.sqrt(sum / buf.length) > 0.02;
  }

  /**
   * Barge-in. The model generates speech faster than real time, so by the time the
   * customer cuts in the response is often already complete and only the buffered
   * audio is still playing: cancelling the response alone isn't enough. So: mute
   * locally first (instant), then cancel, clear the server's output buffer, and
   * truncate the assistant item to what was actually heard so the model's context
   * matches reality.
   */
  private bargeIn() {
    this.interruptAt = performance.now();
    if (this.audio) this.audio.muted = true;
    this.muted = true;
    if (this.responseActive) this.send({ type: "response.cancel" });
    this.send({ type: "output_audio_buffer.clear" });
    if (this.audioItem?.playStart) {
      this.send({
        type: "conversation.item.truncate",
        item_id: this.audioItem.id,
        content_index: 0,
        audio_end_ms: Math.max(0, Math.round(performance.now() - this.audioItem.playStart)),
      });
    }
    this.audioItem = null;
    this.speaking = false;
    for (const id of this.assistantText.keys()) this.interrupted.add(id);
    if (this.lastResponseId) this.cb.onInterrupted(this.lastResponseId);
  }

  private setState(s: VoiceState) {
    this.state = s;
    this.cb.onState(s);
  }

  async connect(sessionId: string): Promise<void> {
    this.sessionId = sessionId;
    this.setState("connecting");
    const secret = await voiceSession(sessionId);
    this.mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.ctx = new AudioContext();
    this.micAnalyser = this.ctx.createAnalyser();
    this.micAnalyser.fftSize = 1024;
    this.ctx.createMediaStreamSource(this.mic).connect(this.micAnalyser);

    const pc = new RTCPeerConnection();
    this.pc = pc;
    this.audio = new Audio();
    this.audio.autoplay = true;
    pc.ontrack = (e) => {
      const stream = e.streams[0];
      this.audio!.srcObject = stream;
      this.outAnalyser = this.ctx!.createAnalyser();
      this.outAnalyser.fftSize = 1024;
      this.ctx!.createMediaStreamSource(stream).connect(this.outAnalyser);
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === "failed" || pc.connectionState === "disconnected") {
        if (!this.closed) this.fail("The voice connection dropped. Switch back to chat or try voice again.");
      }
    };
    pc.addTrack(this.mic.getAudioTracks()[0], this.mic);

    const dc = pc.createDataChannel("oai-events");
    this.dc = dc;
    dc.onopen = () => this.setState("listening");
    dc.onmessage = (e) => {
      try {
        this.handle(JSON.parse(e.data));
      } catch (err) {
        console.warn("bad realtime event", err);
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    const resp = await fetch("https://api.openai.com/v1/realtime/calls", {
      method: "POST",
      body: offer.sdp,
      headers: { Authorization: `Bearer ${secret.client_secret}`, "Content-Type": "application/sdp" },
    });
    if (!resp.ok) throw new Error(`Voice connection was refused (${resp.status}).`);
    await pc.setRemoteDescription({ type: "answer", sdp: await resp.text() });

    // Cost guard for a public demo: hard cap on session length.
    this.timer = window.setTimeout(() => this.disconnect("Voice sessions are capped at " + Math.round(secret.max_session_s / 60) + " minutes in this demo."), secret.max_session_s * 1000);
  }

  private send(evt: Evt) {
    if (this.dc?.readyState === "open") this.dc.send(JSON.stringify(evt));
  }

  private fail(message: string) {
    this.cb.onError(message);
    this.disconnect("error");
    this.setState("error");
  }

  private handle(ev: Evt) {
    switch (ev.type) {
      case "input_audio_buffer.speech_started": {
        if (this.responseActive || this.agentAudible()) this.bargeIn();
        this.cb.onUserSpeechStart(ev.item_id);
        this.setState("user_speaking");
        break;
      }
      case "input_audio_buffer.speech_stopped":
        this.tSpeechStopped = performance.now();
        this.tFirstAudio = 0;
        this.setState("thinking");
        break;
      case "conversation.item.input_audio_transcription.delta": {
        const text = (this.userText.get(ev.item_id) ?? "") + (ev.delta ?? "");
        this.userText.set(ev.item_id, text);
        this.cb.onUserTranscript(ev.item_id, text, false);
        break;
      }
      case "conversation.item.input_audio_transcription.completed": {
        const text = (ev.transcript ?? "").trim();
        this.userText.delete(ev.item_id);
        this.lastUserText = text;
        this.cb.onUserTranscript(ev.item_id, text, true);
        if (text) voiceTranscript(this.sessionId, "user", text);
        break;
      }
      case "response.created":
        this.responseActive = true;
        this.lastResponseId = ev.response.id;
        if (this.muted && this.audio) {
          this.audio.muted = false; // the new answer should be heard
          this.muted = false;
        }
        this.assistantText.set(ev.response.id, "");
        this.cb.onAssistantStart(ev.response.id);
        break;
      case "response.output_item.added":
        if (ev.item?.type === "message" && ev.item?.role === "assistant") {
          this.audioItem = { id: ev.item.id, playStart: 0 };
        }
        break;
      case "response.output_audio_transcript.delta":
      case "response.audio_transcript.delta": {
        const id = ev.response_id;
        if (this.audioItem && !this.audioItem.playStart) this.audioItem.playStart = performance.now();
        this.assistantText.set(id, (this.assistantText.get(id) ?? "") + ev.delta);
        this.cb.onAssistantDelta(id, ev.delta);
        break;
      }
      case "output_audio_buffer.started":
        this.speaking = true;
        if (this.audioItem) this.audioItem.playStart = performance.now();
        if (!this.tFirstAudio) this.tFirstAudio = performance.now();
        this.setState("speaking");
        break;
      case "output_audio_buffer.stopped":
      case "output_audio_buffer.cleared":
        this.speaking = false;
        this.audioItem = null;
        if (this.state === "speaking") this.setState("listening");
        break;
      case "response.function_call_arguments.done": {
        const responseId = ev.response_id;
        this.cb.onToolStart(responseId, ev.call_id, ev.name);
        this.setState("thinking");
        const p = voiceTool(this.sessionId, ev.call_id, ev.name, ev.arguments ?? "{}")
          .then((result) => {
            this.send({ type: "conversation.item.create", item: { type: "function_call_output", call_id: ev.call_id, output: result.output } });
            this.cb.onToolEnd(ev.call_id, result);
          })
          .catch((err: Error) => {
            const output = JSON.stringify({ error: "tool_unavailable", message: `${err.message}. Don't guess; apologise and offer a human.` });
            this.send({ type: "conversation.item.create", item: { type: "function_call_output", call_id: ev.call_id, output } });
            this.cb.onToolEnd(ev.call_id, null, err.message);
          });
        this.pendingTools.set(responseId, [...(this.pendingTools.get(responseId) ?? []), p]);
        break;
      }
      case "response.done": {
        const r = ev.response ?? {};
        const id: string = r.id;
        this.responseActive = false;
        const wasInterrupted = r.status === "cancelled" || this.interrupted.has(id);
        const text = this.assistantText.get(id) ?? "";
        this.assistantText.delete(id);
        this.interrupted.delete(id);
        this.cb.onAssistantDone(id, text, wasInterrupted);
        if (text) voiceTranscript(this.sessionId, "assistant", text, wasInterrupted);
        const now = performance.now();
        voiceUsage(this.sessionId, {
          usage: r.usage ?? {},
          status: r.status ?? "completed",
          user_text: this.lastUserText,
          latency: {
            first_audio_ms: this.tFirstAudio && this.tSpeechStopped ? Math.round(this.tFirstAudio - this.tSpeechStopped) : null,
            response_ms: this.tSpeechStopped ? Math.round(now - this.tSpeechStopped) : null,
          },
        })
          .then((t) => this.cb.onTrace(t as Trace))
          .catch(() => undefined);
        const tools = this.pendingTools.get(id);
        if (tools?.length) {
          this.pendingTools.delete(id);
          Promise.all(tools).then(() => this.send({ type: "response.create" }));
        } else if (this.state === "thinking") {
          this.setState("listening");
        }
        break;
      }
      case "error": {
        const code = ev.error?.code ?? "";
        // Cancel/clear/truncate race the server during a barge-in; "nothing to cancel" is expected.
        const duringBargeIn = performance.now() - this.interruptAt < 2000;
        if (code === "response_cancel_not_active" || duringBargeIn) break;
        console.warn("realtime error", ev.error);
        this.cb.onError(ev.error?.message ?? "Voice error");
        break;
      }
    }
  }

  disconnect(reason = "ended") {
    if (this.closed) return;
    this.closed = true;
    window.clearTimeout(this.timer);
    this.dc?.close();
    this.pc?.getSenders().forEach((s) => s.track?.stop());
    this.pc?.close();
    this.mic?.getTracks().forEach((t) => t.stop());
    if (this.audio) this.audio.srcObject = null;
    this.ctx?.close().catch(() => undefined);
    this.setState("off");
    this.cb.onEnded(reason);
  }
}
