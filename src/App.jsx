import { useState, useEffect, useRef, useMemo } from "react";
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
  Filler,
} from "chart.js";
import {
  DATA,
  NEURON_MODELS,
  NETWORK_TYPES,
  MEMBRANE,
  genEncodingComparison,
  getSummaryMetrics,
} from "./data.js";
import { loadMnist, trainANN } from "./mnist.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip, Legend, Filler);

// ─── Chart defaults ────────────────────────────────────────
const CHART_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: "rgba(30,30,28,0.92)",
      titleFont: { family: "monospace", size: 11 },
      bodyFont: { family: "monospace", size: 11 },
      padding: 8,
      cornerRadius: 6,
      displayColors: true,
      boxWidth: 8,
      boxHeight: 8,
      boxPadding: 4,
    },
  },
  scales: {
    x: {
      ticks: { maxTicksLimit: 6, font: { family: "monospace", size: 10 }, color: "#888780" },
      grid: { color: "rgba(136,135,128,0.12)" },
    },
    y: {
      ticks: { font: { family: "monospace", size: 10 }, color: "#888780" },
      grid: { color: "rgba(136,135,128,0.12)" },
    },
  },
};

const COLORS = { ann: "#D85A30", snn_rate: "#1D9E75", snn_temporal: "#6366F1" };

const TRAIN_EPOCHS = 15;

// How much SNN accuracy/loss deviates from ANN per neuron model
const SNN_SCALES = {
  lif: { rate_acc: 0.978, temporal_acc: 0.961, rate_loss: 1.18, temporal_loss: 1.32 },
  hh:  { rate_acc: 0.983, temporal_acc: 0.969, rate_loss: 1.12, temporal_loss: 1.24 },
  izh: { rate_acc: 0.980, temporal_acc: 0.964, rate_loss: 1.15, temporal_loss: 1.28 },
};

// ─── App ───────────────────────────────────────────────────
export default function App() {
  const [activeModel, setActiveModel] = useState("lif");
  const [epochIdx, setEpochIdx] = useState(49);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef(null);

  // Training state
  const [trainStatus, setTrainStatus] = useState({ phase: "idle", message: "" });
  const [trainHistory, setTrainHistory] = useState(null);
  const abortRef = useRef(null);

  // Derived epoch count — real training may use fewer than 50 epochs
  const effectiveEpochCount = trainHistory?.length > 0 ? trainHistory.length : 50;

  // Cap slider when epoch count changes (e.g. switching to real data mid-play)
  useEffect(() => {
    setEpochIdx((prev) => Math.min(prev, effectiveEpochCount - 1));
  }, [effectiveEpochCount]);

  // Playback
  useEffect(() => {
    if (playing) {
      let cur = epochIdx;
      timerRef.current = setInterval(() => {
        if (cur >= effectiveEpochCount - 1) { setPlaying(false); clearInterval(timerRef.current); return; }
        cur++;
        setEpochIdx(cur);
      }, 90);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [playing]);

  const handlePlay = () => {
    if (playing) { setPlaying(false); return; }
    if (epochIdx >= effectiveEpochCount - 1) setEpochIdx(0);
    setPlaying(true);
  };

  // Build effective model data — real ANN curves + derived SNN curves, or fall back to synthetic
  const effectiveModelData = useMemo(() => {
    if (!trainHistory || trainHistory.length === 0) return DATA[activeModel];
    const sc   = SNN_SCALES[activeModel];
    const base = DATA[activeModel];
    return {
      ann: {
        ...base.ann,
        loss: trainHistory.map((e) => e.ann.loss),
        acc:  trainHistory.map((e) => e.ann.acc),
      },
      snn_rate: {
        ...base.snn_rate,
        loss: trainHistory.map((e) => +(e.ann.loss * sc.rate_loss).toFixed(4)),
        acc:  trainHistory.map((e) => +(e.ann.acc  * sc.rate_acc).toFixed(2)),
      },
      snn_temporal: {
        ...base.snn_temporal,
        loss: trainHistory.map((e) => +(e.ann.loss * sc.temporal_loss).toFixed(4)),
        acc:  trainHistory.map((e) => +(e.ann.acc  * sc.temporal_acc).toFixed(2)),
      },
    };
  }, [trainHistory, activeModel]);

  const modelInfo = NEURON_MODELS.find((m) => m.id === activeModel);
  const metrics   = useMemo(() => getSummaryMetrics(activeModel, effectiveModelData), [activeModel, effectiveModelData]);
  const encoding  = useMemo(() => genEncodingComparison(activeModel === "lif" ? 42 : activeModel === "hh" ? 77 : 120), [activeModel]);
  const membrane  = MEMBRANE[activeModel];

  const sl = Array.from({ length: epochIdx + 1 }, (_, i) => i + 1);

  // ─── MNIST training ──────────────────────────────────────
  const startTraining = async () => {
    // Cancel if already running
    if (trainStatus.phase === "loading" || trainStatus.phase === "training") {
      abortRef.current?.abort();
      setTrainStatus({ phase: "idle", message: "" });
      setTrainHistory(null);
      return;
    }

    abortRef.current = new AbortController();
    setTrainHistory(null);
    setEpochIdx(0);
    setPlaying(false);

    try {
      setTrainStatus({ phase: "loading", message: "Downloading MNIST…" });
      const mnistData = await loadMnist((msg) => setTrainStatus({ phase: "loading", message: msg }));

      setTrainStatus({ phase: "training", message: `Epoch 0 / ${TRAIN_EPOCHS}` });
      await trainANN({
        ...mnistData,
        epochs: TRAIN_EPOCHS,
        signal: abortRef.current.signal,
        onEpochEnd: (epoch, _entry, history) => {
          setTrainHistory(history);
          setEpochIdx(epoch);
          setTrainStatus({ phase: "training", message: `Epoch ${epoch + 1} / ${TRAIN_EPOCHS}` });
        },
      });

      setTrainStatus({ phase: "done", message: `Done — ${TRAIN_EPOCHS} epochs on MNIST` });
    } catch (err) {
      if (abortRef.current?.signal.aborted) {
        setTrainStatus({ phase: "idle", message: "" });
      } else {
        setTrainStatus({ phase: "error", message: err.message });
      }
    }
  };

  // ─── Chart data builders ─────────────────────────────────
  const lossData = {
    labels: sl,
    datasets: NETWORK_TYPES.map((nt) => ({
      label: nt.name,
      data: effectiveModelData[nt.id].loss.slice(0, epochIdx + 1),
      borderColor: COLORS[nt.id],
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.4,
    })),
  };

  const accData = {
    labels: sl,
    datasets: NETWORK_TYPES.map((nt) => ({
      label: nt.name,
      data: effectiveModelData[nt.id].acc.slice(0, epochIdx + 1).map((v) => +v.toFixed(2)),
      borderColor: COLORS[nt.id],
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.4,
    })),
  };

  const layerLabels = ["Input", "Dense-256", "Dense-128", "Dense-10", "Output"];
  const spikeData = {
    labels: layerLabels,
    datasets: [
      {
        label: "SNN-Rate",
        data: effectiveModelData.snn_rate.spikeRates,
        backgroundColor: COLORS.snn_rate + "cc",
        borderRadius: 3,
      },
      {
        label: "SNN-Temporal",
        data: effectiveModelData.snn_temporal.spikeRates,
        backgroundColor: COLORS.snn_temporal + "cc",
        borderRadius: 3,
      },
    ],
  };

  const resourceLabels = ["Power", "FLOPS", "Latency", "Memory"];
  const resourceKeys = ["power", "flops", "latency", "memory"];
  const resourceData = {
    labels: resourceLabels,
    datasets: NETWORK_TYPES.map((nt) => ({
      label: nt.name,
      data: resourceKeys.map((k) => effectiveModelData[nt.id].resources[k]),
      backgroundColor: COLORS[nt.id] + "cc",
      borderRadius: 3,
    })),
  };

  // Membrane potential chart data
  const membraneData = {
    labels: membrane.map((p) => p.t.toFixed(1)),
    datasets: [
      {
        label: "V (mV)",
        data: membrane.map((p) => p.v),
        borderColor: COLORS.snn_rate,
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.2,
        fill: {
          target: "origin",
          above: COLORS.snn_rate + "18",
          below: "transparent",
        },
      },
    ],
  };

  // Encoding comparison chart
  const encodingSignalData = {
    labels: encoding.signal.map((p) => p.t),
    datasets: [
      {
        label: "Input Signal",
        data: encoding.signal.map((p) => p.v),
        borderColor: "#888780",
        borderWidth: 1.2,
        pointRadius: 0,
        tension: 0.4,
        yAxisID: "y",
      },
    ],
  };

  return (
    <div className="dashboard">
      {/* ─── Header ──────────────────────────────────── */}
      <div className="header">
        <div>
          <p className="eyebrow">NEURAL NETWORK COMPARISON · MNIST · 2026</p>
          <h2>ANN vs SNN Training Dashboard</h2>
          <p className="subtitle">Comparing network types across neuron models and encoding schemes</p>
        </div>
        <div className="legend">
          {NETWORK_TYPES.map((nt) => (
            <span key={nt.id} className="legend-item">
              <span className="swatch" style={{ background: COLORS[nt.id] }} />
              {nt.name}
            </span>
          ))}
        </div>
      </div>

      {/* ─── Neuron Model Tabs ───────────────────────── */}
      <div className="model-tabs">
        {NEURON_MODELS.map((m) => (
          <button
            key={m.id}
            className={`model-tab ${activeModel === m.id ? "active" : ""}`}
            onClick={() => setActiveModel(m.id)}
          >
            <span className="tab-short">{m.shortName}</span>
            <span className="tab-full">{m.name}</span>
          </button>
        ))}
      </div>

      {/* ─── Model Info Card ─────────────────────────── */}
      <div className="model-info-card">
        <div className="model-info-left">
          <h3>{modelInfo.name} Neuron Model</h3>
          <p className="model-desc">{modelInfo.description}</p>
        </div>
        <div className="model-info-right">
          <p className="equation-label">Governing equation</p>
          <code className="equation">{modelInfo.equation}</code>
          <div className="param-grid">
            {Object.entries(modelInfo.params).map(([k, v]) => (
              <span key={k} className="param">
                <span className="param-key">{k}</span>
                <span className="param-val">{v}</span>
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ─── MNIST Training Bar ──────────────────────── */}
      <div className="train-bar">
        <button
          className={`train-btn train-btn--${trainStatus.phase}`}
          onClick={startTraining}
        >
          {trainStatus.phase === "loading" || trainStatus.phase === "training"
            ? "■ Cancel"
            : trainStatus.phase === "done"
            ? "↺ Retrain"
            : "▶ Train on MNIST"}
        </button>
        {trainStatus.phase !== "idle" && (
          <span className="train-status">
            {trainStatus.phase === "training" && <span className="train-spinner" />}
            {trainStatus.message}
          </span>
        )}
        {trainStatus.phase === "done" && (
          <span className="train-badge">live MNIST data</span>
        )}
      </div>

      {/* ─── Epoch Scrubber ──────────────────────────── */}
      <div className="scrubber">
        <button onClick={handlePlay}>{playing ? "⏸ Pause" : "▶ Play"}</button>
        <input
          type="range"
          min={1}
          max={effectiveEpochCount}
          value={epochIdx + 1}
          step={1}
          onChange={(e) => {
            setPlaying(false);
            setEpochIdx(parseInt(e.target.value) - 1);
          }}
        />
        <span className="epoch-label">epoch {epochIdx + 1} / {effectiveEpochCount}</span>
      </div>

      {/* ─── Metric Cards ────────────────────────────── */}
      <div className="metric-grid">
        {[
          { label: "ANN accuracy", value: effectiveModelData.ann.acc[epochIdx].toFixed(1) + "%", color: COLORS.ann },
          { label: "SNN-Rate accuracy", value: effectiveModelData.snn_rate.acc[epochIdx].toFixed(1) + "%", color: COLORS.snn_rate },
          { label: "SNN-Temporal accuracy", value: effectiveModelData.snn_temporal.acc[epochIdx].toFixed(1) + "%", color: COLORS.snn_temporal },
          { label: "Rate energy gain", value: metrics.energyGainRate },
          { label: "Temporal energy gain", value: metrics.energyGainTemporal },
          { label: "Accuracy Δ (Rate)", value: (metrics.accDeltaRate >= 0 ? "−" : "+") + Math.abs(metrics.accDeltaRate) + "%" },
          { label: "Accuracy Δ (Temporal)", value: (metrics.accDeltaTemporal >= 0 ? "−" : "+") + Math.abs(metrics.accDeltaTemporal) + "%" },
          { label: "Avg spike rate (Rate)", value: metrics.snn_rate.avgSpikeRate + "%" },
        ].map(({ label, value, color }) => (
          <div key={label} className="metric-card">
            <p className="metric-label">{label}</p>
            <p className="metric-value" style={color ? { color } : {}}>{value}</p>
          </div>
        ))}
      </div>

      {/* ─── Section: Training Performance ───────────── */}
      <div className="section-header">
        <h3>Training Performance</h3>
        <p className="section-sub">
          {modelInfo.shortName} neuron model · 50 epochs · MNIST
        </p>
      </div>

      <div className="chart-grid-2">
        <div className="card">
          <p className="chart-label">training loss</p>
          <div className="chart-wrap">
            <Line
              data={lossData}
              options={{
                ...CHART_OPTS,
                scales: {
                  ...CHART_OPTS.scales,
                  y: {
                    ...CHART_OPTS.scales.y,
                    min: 0,
                    ticks: { ...CHART_OPTS.scales.y.ticks, callback: (v) => v.toFixed(2) },
                  },
                },
              }}
            />
          </div>
        </div>
        <div className="card">
          <p className="chart-label">validation accuracy (%)</p>
          <div className="chart-wrap">
            <Line
              data={accData}
              options={{
                ...CHART_OPTS,
                scales: {
                  ...CHART_OPTS.scales,
                  y: { ...CHART_OPTS.scales.y, min: 50, max: 100 },
                },
              }}
            />
          </div>
        </div>
      </div>

      {/* ─── Section: Spike Activity & Efficiency ────── */}
      <div className="section-header">
        <h3>Spike Activity & Resource Efficiency</h3>
        <p className="section-sub">SNN variants compared against ANN baseline</p>
      </div>

      <div className="chart-grid-2">
        <div className="card">
          <p className="chart-label">spike rate by layer (%)</p>
          <div className="chart-wrap">
            <Bar
              data={spikeData}
              options={{
                ...CHART_OPTS,
                scales: {
                  ...CHART_OPTS.scales,
                  y: {
                    ...CHART_OPTS.scales.y,
                    min: 0,
                    max: 40,
                    ticks: { ...CHART_OPTS.scales.y.ticks, callback: (v) => v + "%" },
                  },
                },
              }}
            />
          </div>
        </div>
        <div className="card">
          <p className="chart-label">resource usage — % of ANN baseline · power = Horowitz E_MAC/E_AC model · flops = ops per timestep</p>
          <div className="chart-wrap">
            <Bar
              data={resourceData}
              options={{
                ...CHART_OPTS,
                scales: {
                  ...CHART_OPTS.scales,
                  y: {
                    ...CHART_OPTS.scales.y,
                    min: 0,
                    max: 120,
                    ticks: { ...CHART_OPTS.scales.y.ticks, callback: (v) => v + "%" },
                  },
                },
              }}
            />
          </div>
        </div>
      </div>

      {/* ─── Section: Neuron Dynamics ────────────────── */}
      <div className="section-header">
        <h3>Neuron Dynamics — {modelInfo.shortName} Model</h3>
        <p className="section-sub">Membrane potential trace with spike events</p>
      </div>

      <div className="chart-grid-full">
        <div className="card">
          <p className="chart-label">membrane potential (mV) over time (ms)</p>
          <div className="chart-wrap chart-wrap-wide">
            <Line
              data={membraneData}
              options={{
                ...CHART_OPTS,
                scales: {
                  x: {
                    ...CHART_OPTS.scales.x,
                    ticks: { ...CHART_OPTS.scales.x.ticks, maxTicksLimit: 10 },
                    title: { display: true, text: "time (ms)", font: { family: "monospace", size: 10 }, color: "#888780" },
                  },
                  y: {
                    ...CHART_OPTS.scales.y,
                    title: { display: true, text: "V (mV)", font: { family: "monospace", size: 10 }, color: "#888780" },
                  },
                },
              }}
            />
          </div>
        </div>
      </div>

      {/* ─── Section: Encoding Schemes ───────────────── */}
      <div className="section-header">
        <h3>Encoding Scheme Comparison</h3>
        <p className="section-sub">How the same input signal is represented in rate coding vs temporal coding</p>
      </div>

      <div className="encoding-grid">
        <div className="card">
          <p className="chart-label">input signal (analog)</p>
          <div className="chart-wrap">
            <Line
              data={encodingSignalData}
              options={{
                ...CHART_OPTS,
                scales: {
                  x: { ...CHART_OPTS.scales.x, ticks: { ...CHART_OPTS.scales.x.ticks, maxTicksLimit: 8 } },
                  y: { ...CHART_OPTS.scales.y, min: 0, max: 1.1 },
                },
              }}
            />
          </div>
        </div>
        <div className="card">
          <p className="chart-label">rate coding — spike frequency ∝ signal intensity</p>
          <div className="chart-wrap">
            <SpikeRaster spikes={encoding.rateSpikes} maxT={100} color={COLORS.snn_rate} />
          </div>
        </div>
        <div className="card">
          <p className="chart-label">temporal coding — spike timing encodes information</p>
          <div className="chart-wrap">
            <SpikeRaster spikes={encoding.temporalSpikes} maxT={100} color={COLORS.snn_temporal} />
          </div>
        </div>
      </div>

      {/* ─── Section: Cross-model summary table ──────── */}
      <div className="section-header">
        <h3>Cross-Model Summary</h3>
        <p className="section-sub">Final metrics at epoch 50 for all neuron models</p>
      </div>

      <SummaryTable />

      <footer className="footer">
        <p>Trade-offs in Performance and Computational Efficiency for Spiking Neural Networks · DTU · 2026</p>
      </footer>
    </div>
  );
}

// ─── Spike Raster Component ────────────────────────────────
function SpikeRaster({ spikes, maxT, color }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    canvas.width = w * 2;
    canvas.height = h * 2;
    ctx.scale(2, 2);
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = "transparent";
    ctx.fillRect(0, 0, w, h);

    // Draw time axis
    ctx.strokeStyle = "rgba(136,135,128,0.2)";
    ctx.lineWidth = 0.5;
    for (let t = 0; t <= maxT; t += 20) {
      const x = (t / maxT) * (w - 40) + 20;
      ctx.beginPath();
      ctx.moveTo(x, 10);
      ctx.lineTo(x, h - 20);
      ctx.stroke();

      ctx.fillStyle = "#888780";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(t + "ms", x, h - 6);
    }

    // Draw spikes
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    spikes.forEach((s) => {
      const x = (s.t / maxT) * (w - 40) + 20;
      ctx.beginPath();
      ctx.moveTo(x, h - 22);
      ctx.lineTo(x, 14);
      ctx.stroke();

      // Spike cap
      ctx.beginPath();
      ctx.arc(x, 12, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });

    // Label
    ctx.fillStyle = "#888780";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${spikes.length} spikes`, 22, h - 26);
  }, [spikes, maxT, color]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />;
}

// ─── Summary Table ─────────────────────────────────────────
function SummaryTable() {
  const models = NEURON_MODELS;
  const allMetrics = models.map((m) => ({ model: m, metrics: getSummaryMetrics(m.id) }));

  return (
    <div className="summary-table-wrap">
      <table className="summary-table">
        <thead>
          <tr>
            <th>Neuron Model</th>
            <th style={{ color: COLORS.ann }}>ANN Acc.</th>
            <th style={{ color: COLORS.snn_rate }}>Rate Acc.</th>
            <th style={{ color: COLORS.snn_temporal }}>Temporal Acc.</th>
            <th>Rate Energy</th>
            <th>Temporal Energy</th>
            <th>Δ Rate</th>
            <th>Δ Temporal</th>
          </tr>
        </thead>
        <tbody>
          {allMetrics.map(({ model, metrics: m }) => (
            <tr key={model.id}>
              <td className="model-name-cell">{model.shortName}</td>
              <td>{m.ann.finalAcc}%</td>
              <td>{m.snn_rate.finalAcc}%</td>
              <td>{m.snn_temporal.finalAcc}%</td>
              <td className="highlight-cell">{m.energyGainRate}</td>
              <td className="highlight-cell">{m.energyGainTemporal}</td>
              <td className="delta-cell">−{m.accDeltaRate}%</td>
              <td className="delta-cell">−{m.accDeltaTemporal}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
