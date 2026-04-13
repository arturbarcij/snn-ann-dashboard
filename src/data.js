// ─── Seeded RNG ─────────────────────────────────────────────
export function rng(seed) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

const N = 50;
export const epochs = Array.from({ length: N }, (_, i) => i + 1);

// ─── Training curves generator ──────────────────────────────
// Each neuron model + network type combo gets unique but realistic curves
function genLoss(seed, decayRate, baseline, noiseAmp) {
  const r = rng(seed);
  return epochs.map((e) => {
    const t = e / N;
    return Math.max(baseline * 0.8, 2.2 * Math.exp(-decayRate * t) + baseline + (r() - 0.5) * noiseAmp);
  });
}

function genAcc(seed, maxAcc, riseRate, noiseAmp) {
  const r = rng(seed);
  return epochs.map((e) => {
    const t = e / N;
    return Math.min(maxAcc, Math.max(10, (maxAcc - 0.5) * (1 - Math.exp(-riseRate * t)) + (r() - 0.5) * noiseAmp));
  });
}

// ─── Neuron model configs ───────────────────────────────────
// For each neuron model, define how the 3 network types perform.
// ANN is the same baseline across all (since ANN doesn't use spiking neuron models),
// but SNN-Rate and SNN-Temporal differ per neuron model.

export const NEURON_MODELS = [
  {
    id: "lif",
    name: "Leaky Integrate-and-Fire",
    shortName: "LIF",
    description: "Simplest biologically-inspired spiking neuron. Membrane potential leaks over time and fires when threshold is reached. Computationally efficient, widely used in neuromorphic hardware.",
    equation: "τ_m · dV/dt = -(V - V_rest) + R·I(t)",
    params: { threshold: "−55 mV", rest: "−70 mV", tau: "20 ms", reset: "−70 mV" },
  },
  {
    id: "hh",
    name: "Hodgkin-Huxley",
    shortName: "H-H",
    description: "Biophysically detailed model with voltage-gated ion channels (Na⁺, K⁺). Captures realistic action potential shapes but is computationally expensive.",
    equation: "C_m · dV/dt = -g_Na·m³h(V-E_Na) - g_K·n⁴(V-E_K) - g_L(V-E_L) + I(t)",
    params: { "g_Na": "120 mS/cm²", "g_K": "36 mS/cm²", "C_m": "1 µF/cm²", "E_Na": "+50 mV" },
  },
  {
    id: "izh",
    name: "Izhikevich",
    shortName: "Izhikevich",
    description: "Efficient 2-variable model that reproduces most known spiking patterns (regular, bursting, chattering, etc.). Good balance between biological plausibility and computational cost.",
    equation: "dv/dt = 0.04v² + 5v + 140 − u + I;  du/dt = a(bv − u);  if v≥30: v←c, u←u+d",
    params: { a: "0.02", b: "0.2", c: "−65 mV", d: "8" },
  },
];

export const NETWORK_TYPES = [
  { id: "ann", name: "ANN", color: "#D85A30", label: "Artificial Neural Network" },
  { id: "snn_rate", name: "SNN-Rate", color: "#1D9E75", label: "SNN · Rate Coding" },
  { id: "snn_temporal", name: "SNN-Temporal", color: "#6366F1", label: "SNN · Temporal Coding" },
];

// ─── Energy model ───────────────────────────────────────────
// Matches the actual MLP trained in mnist.js: 784 → 256 → 128 → 10
// LAYER_MACS[i] = MACs for the weight matrix FROM layer i-1 TO layer i
//   Layer 0: Input (784)    — no weights
//   Layer 1: Dense-256      — 784 × 256 = 200,704
//   Layer 2: Dense-128      — 256 × 128 = 32,768
//   Layer 3: Dense-10       — 128 × 10  = 1,280
//   Layer 4: Output (10)    — no weights (softmax only)
const LAYER_MACS = [0, 200_704, 32_768, 1_280, 0];
const ANN_TOTAL_MACS = 234_752; // sum of non-zero entries above

const E_MAC = 4.6; // pJ — multiply-accumulate, 45 nm CMOS (Horowitz 2014)
const E_AC  = 0.9; // pJ — accumulate only (spike-triggered addition)
const T_SNN = 10;  // timesteps per inference (rate coding)

// spikeRates[i] = % of neurons active at layer i (post-activation sparsity)
// For weight layer i, the pre-synaptic spike rate is spikeRates[i-1].
// power  = SNN energy as % of ANN energy  (E_MAC vs E_AC weighted)
// flops  = per-timestep SNN ops as % of ANN MACs  (operation count only)
// latency and memory remain synthetic estimates
export function computeSnnResources(spikeRates, latency, memory) {
  let snnEnergy = 0;
  let snnOps    = 0;
  for (let i = 1; i < LAYER_MACS.length; i++) {
    if (LAYER_MACS[i] === 0) continue;
    const preRate  = spikeRates[i - 1] / 100;
    snnEnergy += LAYER_MACS[i] * preRate * T_SNN * E_AC;
    snnOps    += LAYER_MACS[i] * preRate; // per-timestep ops (no E weighting)
  }
  const annEnergy = ANN_TOTAL_MACS * E_MAC;
  return {
    power:   +((snnEnergy / annEnergy) * 100).toFixed(1), // % of ANN energy
    flops:   +((snnOps    / ANN_TOTAL_MACS) * 100).toFixed(1), // % of ANN MACs per timestep
    latency,
    memory,
  };
}

// ─── Generate all data ──────────────────────────────────────
// Structure: data[modelId][networkTypeId] = { loss, acc, spikeRates, resources, membranePotential }

export const DATA = {};

// --- ANN baseline (same for all neuron model tabs) ---
const annLoss = genLoss(99, 5.0, 0.058, 0.018);
const annAcc = genAcc(13, 98.2, 6.0, 0.7);
const annBaseline = {
  loss: annLoss,
  acc: annAcc,
  spikeRates: null, // ANNs don't have spike rates
  resources: { power: 100, flops: 100, latency: 60, memory: 100 },
};

// --- LIF Model ---
// Simplest neuron: clean ReLU-like threshold → moderate sparsity
// Rate: ~18% input activation  →  ~3× energy gain vs ANN
// Temporal: ~10% input activation  →  ~5.4× energy gain
DATA.lif = {
  ann: annBaseline,
  snn_rate: {
    loss: genLoss(42, 4.2, 0.108, 0.025),
    acc: genAcc(7, 96.8, 5.5, 0.9),
    spikeRates: [18, 10, 7, 4, 2],
    resources: computeSnnResources([18, 10, 7, 4, 2], 85, 35),
  },
  snn_temporal: {
    loss: genLoss(55, 3.6, 0.145, 0.032),
    acc: genAcc(22, 95.4, 4.8, 1.1),
    spikeRates: [10, 6, 4, 2, 1],
    resources: computeSnnResources([10, 6, 4, 2, 1], 100, 28),
  },
};

// --- Hodgkin-Huxley Model ---
// Complex ion-channel dynamics → higher baseline activation, less sparsity
// Rate: ~20% input activation  →  ~2.7× energy gain
// Temporal: ~10% input activation  →  ~5.3× energy gain
DATA.hh = {
  ann: annBaseline,
  snn_rate: {
    loss: genLoss(101, 3.5, 0.14, 0.03),
    acc: genAcc(33, 97.2, 4.5, 0.8),
    spikeRates: [20, 12, 8, 5, 2],
    resources: computeSnnResources([20, 12, 8, 5, 2], 92, 52),
  },
  snn_temporal: {
    loss: genLoss(77, 3.0, 0.18, 0.038),
    acc: genAcc(44, 96.1, 4.0, 1.2),
    spikeRates: [10, 7, 5, 3, 1],
    resources: computeSnnResources([10, 7, 5, 3, 1], 100, 45),
  },
};

// --- Izhikevich Model ---
// Efficient 2-variable model → slightly lower spike rates than LIF
// Rate: ~16% input activation  →  ~3.4× energy gain
// Temporal: ~9% input activation  →  ~6× energy gain
DATA.izh = {
  ann: annBaseline,
  snn_rate: {
    loss: genLoss(120, 4.5, 0.095, 0.022),
    acc: genAcc(50, 97.5, 5.8, 0.75),
    spikeRates: [16, 10, 7, 4, 2],
    resources: computeSnnResources([16, 10, 7, 4, 2], 80, 38),
  },
  snn_temporal: {
    loss: genLoss(88, 3.8, 0.13, 0.028),
    acc: genAcc(61, 96.2, 5.0, 1.0),
    spikeRates: [9, 6, 3, 2, 1],
    resources: computeSnnResources([9, 6, 3, 2, 1], 95, 30),
  },
};

// ─── Membrane Potential Simulations ─────────────────────────
// Pre-computed time series showing neuron dynamics for each model

function genMembraneLIF(seed) {
  const r = rng(seed);
  const steps = 200;
  const dt = 0.5; // ms
  const tau = 20;
  const vRest = -70;
  const vThresh = -55;
  const vReset = -70;
  let v = vRest;
  const result = [];
  for (let i = 0; i < steps; i++) {
    const I = 18 + (r() - 0.5) * 12;
    v += dt * (-(v - vRest) + I) / tau;
    if (v >= vThresh) {
      result.push({ t: i * dt, v: 20 }); // spike
      v = vReset;
    } else {
      result.push({ t: i * dt, v });
    }
  }
  return result;
}

function genMembraneHH(seed) {
  const r = rng(seed);
  const steps = 200;
  const dt = 0.05;
  const cm = 1.0;
  const gNa = 120, gK = 36, gL = 0.3;
  const eNa = 50, eK = -77, eL = -54.4;
  let v = -65, m = 0.05, h = 0.6, n = 0.32;

  const alphaM = (v) => (v + 40) === 0 ? 1 : 0.1 * (v + 40) / (1 - Math.exp(-(v + 40) / 10));
  const betaM = (v) => 4 * Math.exp(-(v + 65) / 18);
  const alphaH = (v) => 0.07 * Math.exp(-(v + 65) / 20);
  const betaH = (v) => 1 / (1 + Math.exp(-(v + 35) / 10));
  const alphaN = (v) => (v + 55) === 0 ? 0.1 : 0.01 * (v + 55) / (1 - Math.exp(-(v + 55) / 10));
  const betaN = (v) => 0.125 * Math.exp(-(v + 65) / 80);

  const result = [];
  for (let i = 0; i < steps; i++) {
    const I = 10 + (r() - 0.5) * 6;
    const iNa = gNa * m * m * m * h * (v - eNa);
    const iK = gK * n * n * n * n * (v - eK);
    const iL = gL * (v - eL);
    // Compute all derivatives at current state before updating anything
    const dv = dt * (I - iNa - iK - iL) / cm;
    const dm = dt * (alphaM(v) * (1 - m) - betaM(v) * m);
    const dh = dt * (alphaH(v) * (1 - h) - betaH(v) * h);
    const dn = dt * (alphaN(v) * (1 - n) - betaN(v) * n);
    v += dv;
    m += dm;
    h += dh;
    n += dn;
    m = Math.max(0, Math.min(1, m));
    h = Math.max(0, Math.min(1, h));
    n = Math.max(0, Math.min(1, n));
    if (i % 4 === 0) result.push({ t: i * dt, v: Math.max(-80, Math.min(50, v)) });
  }
  return result;
}

function genMembraneIzh(seed) {
  const r = rng(seed);
  const steps = 200;
  const dt = 0.5;
  const a = 0.02, b = 0.2, c = -65, d = 8;
  let v = -65, u = b * v;
  const result = [];
  for (let i = 0; i < steps; i++) {
    const I = 14 + (r() - 0.5) * 8;
    // Compute both derivatives at current state before updating
    const dv = dt * (0.04 * v * v + 5 * v + 140 - u + I);
    const du = dt * a * (b * v - u);
    v += dv;
    u += du;
    if (v >= 30) {
      result.push({ t: i * dt, v: 30 });
      v = c;
      u += d;
    } else {
      result.push({ t: i * dt, v: Math.max(-80, Math.min(30, v)) });
    }
  }
  return result;
}

export const MEMBRANE = {
  lif: genMembraneLIF(42),
  hh: genMembraneHH(77),
  izh: genMembraneIzh(120),
};

// ─── Encoding scheme comparison data ────────────────────────
// Shows how rate coding vs temporal coding represent the same input signal

export function genEncodingComparison(seed) {
  const r = rng(seed);
  const steps = 100;
  const dt = 1; // ms

  // Input signal (analog)
  const signal = [];
  for (let i = 0; i < steps; i++) {
    signal.push({ t: i * dt, v: 0.5 + 0.4 * Math.sin(2 * Math.PI * i / 40) + (r() - 0.5) * 0.08 });
  }

  // Rate coding: spike probability proportional to signal value
  const r2 = rng(seed + 100);
  const rateSpikes = [];
  for (let i = 0; i < steps; i++) {
    if (r2() < signal[i].v * 0.4) {
      rateSpikes.push({ t: i * dt, v: 1 });
    }
  }

  // Temporal coding: time-to-first-spike inversely proportional to signal strength
  const r3 = rng(seed + 200);
  const temporalSpikes = [];
  for (let w = 0; w < steps; w += 20) {
    const windowSignal = signal.slice(w, w + 20);
    if (windowSignal.length > 0) {
      const avgVal = windowSignal.reduce((a, b) => a + b.v, 0) / windowSignal.length;
      const delay = Math.round((1 - avgVal) * 15) + Math.round((r3() - 0.5) * 2);
      const spikeTime = w + Math.max(0, Math.min(19, delay));
      temporalSpikes.push({ t: spikeTime * dt, v: 1 });
    }
  }

  return { signal, rateSpikes, temporalSpikes };
}

// ─── Summary metrics per model ──────────────────────────────
// Pass overrideData to use real training data instead of the synthetic DATA.
export function getSummaryMetrics(modelId, overrideData) {
  const d    = overrideData || DATA[modelId];
  const last = d.ann.acc.length - 1;
  return {
    ann: {
      finalAcc:  d.ann.acc[last].toFixed(1),
      finalLoss: d.ann.loss[last].toFixed(3),
      power:     d.ann.resources.power,
    },
    snn_rate: {
      finalAcc:     d.snn_rate.acc[last].toFixed(1),
      finalLoss:    d.snn_rate.loss[last].toFixed(3),
      avgSpikeRate: (d.snn_rate.spikeRates.reduce((a, b) => a + b, 0) / d.snn_rate.spikeRates.length).toFixed(1),
      power:        d.snn_rate.resources.power,
    },
    snn_temporal: {
      finalAcc:     d.snn_temporal.acc[last].toFixed(1),
      finalLoss:    d.snn_temporal.loss[last].toFixed(3),
      avgSpikeRate: (d.snn_temporal.spikeRates.reduce((a, b) => a + b, 0) / d.snn_temporal.spikeRates.length).toFixed(1),
      power:        d.snn_temporal.resources.power,
    },
    energyGainRate:     (100 / d.snn_rate.resources.power).toFixed(2) + "×",
    energyGainTemporal: (100 / d.snn_temporal.resources.power).toFixed(2) + "×",
    accDeltaRate:     (d.ann.acc[last] - d.snn_rate.acc[last]).toFixed(1),
    accDeltaTemporal: (d.ann.acc[last] - d.snn_temporal.acc[last]).toFixed(1),
  };
}
