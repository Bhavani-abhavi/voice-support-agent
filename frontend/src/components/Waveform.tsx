import { useEffect, useRef } from "react";

/** Live waveform for the mic (customer) and the remote stream (agent). */
export function Waveform({ mic, out, active }: { mic?: AnalyserNode; out?: AnalyserNode; active: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const g = canvas.getContext("2d")!;
    let raf = 0;
    const buf = new Uint8Array(512);
    const styles = getComputedStyle(document.documentElement);
    const userColor = styles.getPropertyValue("--user-wave").trim() || "#4f46e5";
    const agentColor = styles.getPropertyValue("--agent-wave").trim() || "#10b981";
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const line = (node: AnalyserNode | undefined, color: string) => {
        g.beginPath();
        g.lineWidth = 2;
        g.strokeStyle = color;
        if (!node || !active) {
          g.moveTo(0, h / 2); g.lineTo(w, h / 2);
        } else {
          node.getByteTimeDomainData(buf);
          for (let i = 0; i < buf.length; i++) {
            const x = (i / (buf.length - 1)) * w;
            const y = h / 2 + ((buf[i] - 128) / 128) * (h / 2) * 1.6;
            i ? g.lineTo(x, y) : g.moveTo(x, y);
          }
        }
        g.stroke();
      };
      g.globalAlpha = 0.9; line(out, agentColor);
      g.globalAlpha = 0.9; line(mic, userColor);
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [mic, out, active]);
  return <canvas ref={ref} className="wave" aria-label="Audio waveform" />;
}
