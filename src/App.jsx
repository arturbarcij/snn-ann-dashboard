import { useState, useEffect, useRef } from "react";
import { Line, Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
  Legend
);

const N = 50;
const epochs = Array.from({ length: N }, (_, i) => i + 1);

function rng(seed) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

const r1 = rng(42), r2 = rng(99), r3 = rng(7), r4 = rng(13);

const snnLoss = epochs.map((e) => {
  const t = e / N;
  return Math.max(0.06, 2.1 * Math.exp(-4.2 * t) + 0.108 + (r1() - 0.5) * 0.025);
});
const annLoss = epochs.map((e) => {
  const t = e / N;
  return Math.max(0.03, 2.35 * Math.exp(-5 * t) + 0.058 + (r2() - 0.5) * 0.018);
});
const snnAcc = epochs.map((e) => {
  const t = e / N;
  return Math.min(96.8, Math.max(10, 96.2 * (1 - Math.exp(-5.5 * t)) + (r3() - 0.5) * 0.9));
});
const annAcc = epochs.map((e) => {
  const t = e / N;
  return Math.min(98.2, Math.max(10, 97.8 * (1 - Math.exp(-6 * t)) + (r4() - 0.5) * 0.7));
});

const CHART_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { maxTicksLimit: 6, font: { family: "monospace", size: 10 }, color: "#888780" }, grid: { color: "rgba(136,135,128,0.15)" } },
    y: { ticks: { font: { family: "monospace", size: 10 }, color: "#888780" }, grid: { color: "rgba(136,135,128,0.15)" } },
  },
};

export default function App() {
  const [epochIdx, setEpochIdx] = useState(49);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef(null);

  useEffect(() => {
    if (playing) {
      let cur = epochIdx;
      timerRef.current = setInterval(() => {
        if (cur >= 49) { setPlaying(false); clearInterval(timerRef.current); return; }
        cur++;
        setEpochIdx(cur);
      }, 90);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [playing]);

  const sl = epochs.slice(0, epochIdx + 1);

  const lossData = {
    labels: sl,
    datasets: [
      { label: "SNN", data: snnLoss.slice(0, epochIdx + 1), borderColor: "#1D9E75", borderWidth: 1.5, pointRadius: 0, tension: 0.4 },
      { label: "ANN", data: annLoss.slice(0, epochIdx + 1), borderColor: "#D85A30", borderWidth: 1.5, pointRadius: 0, tension: 0.4 },
    ],
  };

  const accData = {
    labels: sl,
    datasets: [
      { label: "SNN", data: snnAcc.slice(0, epochIdx + 1).map(v => +v.toFixed(2)), borderColor: "#1D9E75", borderWidth: 1.5, pointRadius: 0, tension: 0.4 },
      { label: "ANN", data: annAcc.slice(0, epochIdx + 1).map(v => +v.toFixed(2)), borderColor: "#D85A30", borderWidth: 1.5, pointRadius: 0, tension: 0.4 },
    ],
  };

  const spikeData = {
    labels: ["Input", "Conv1", "Conv2", "FC1", "FC2", "Output"],
    datasets: [{ label: "Spike rate", data: [32, 18, 12, 9, 7, 5], backgroundColor: "#1D9E75", borderRadius: 3 }],
  };

  const effData = {
    labels: ["Power", "FLOPS", "Latency"],
    datasets: [
      { label: "ANN", data: [100, 100, 67], backgroundColor: "#D85A30", borderRadius: 3 },
      { label: "SNN", data: [26, 26, 100], backgroundColor: "#1D9E75", borderRadius: 3 },
    ],
  };

  const handlePlay = () => {
    if (playing) { setPlaying(false); return; }
    if (epochIdx >= 49) setEpochIdx(0);
    setPlaying(true);
  };

  return (
    <div className="dashboard">
      <div className="header">
        <div>
          <p className="eyebrow">EXPERIMENT · ANN vs SNN · MNIST</p>
          <h2>Training dashboard</h2>
        </div>
        <div className="legend">
          <span className="legend-item"><span className="swatch snn" />SNN</span>
          <span className="legend-item"><span className="swatch ann" />ANN</span>
        </div>
      </div>

      <div className="scrubber">
        <button onClick={handlePlay}>{playing ? "⏸ Pause" : "▶ Play"}</button>
        <input type="range" min={1} max={50} value={epochIdx + 1} step={1}
          onChange={e => { setPlaying(false); setEpochIdx(parseInt(e.target.value) - 1); }} />
        <span className="epoch-label">epoch {epochIdx + 1}</span>
      </div>

      <div className="metric-grid">
        {[
          { label: "SNN val acc", value: snnAcc[epochIdx].toFixed(1) + "%", color: "#1D9E75" },
          { label: "ANN val acc", value: annAcc[epochIdx].toFixed(1) + "%", color: "#D85A30" },
          { label: "Energy gain", value: "3.86×", color: undefined },
          { label: "Avg spike rate", value: "14.3%", color: undefined },
        ].map(({ label, value, color }) => (
          <div key={label} className="metric-card">
            <p className="metric-label">{label}</p>
            <p className="metric-value" style={color ? { color } : {}}>{value}</p>
          </div>
        ))}
      </div>

      <div className="chart-grid-2">
        <div className="card"><p className="chart-label">training loss</p><div className="chart-wrap"><Line data={lossData} options={{ ...CHART_OPTS, scales: { ...CHART_OPTS.scales, y: { ...CHART_OPTS.scales.y, min: 0, ticks: { ...CHART_OPTS.scales.y.ticks, callback: v => v.toFixed(2) } } } }} /></div></div>
        <div className="card"><p className="chart-label">validation accuracy (%)</p><div className="chart-wrap"><Line data={accData} options={{ ...CHART_OPTS, scales: { ...CHART_OPTS.scales, y: { ...CHART_OPTS.scales.y, min: 50, max: 100 } } }} /></div></div>
      </div>

      <div className="chart-grid-3-2">
        <div className="card"><p className="chart-label">spike rate by layer</p><div className="chart-wrap"><Bar data={spikeData} options={{ ...CHART_OPTS, scales: { ...CHART_OPTS.scales, y: { ...CHART_OPTS.scales.y, min: 0, max: 40, ticks: { ...CHART_OPTS.scales.y.ticks, callback: v => v + "%" } } } }} /></div></div>
        <div className="card"><p className="chart-label">resource usage — % of max</p><div className="chart-wrap"><Bar data={effData} options={{ ...CHART_OPTS, scales: { ...CHART_OPTS.scales, y: { ...CHART_OPTS.scales.y, min: 0, max: 120, ticks: { ...CHART_OPTS.scales.y.ticks, callback: v => v + "%" } } } }} /></div></div>
      </div>
    </div>
  );
}
