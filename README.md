# SNN vs ANN · Training Dashboard  *(Experimental)*

An **experimental** interactive dashboard for exploring trade-offs between Spiking Neural Networks (SNN) and Artificial Neural Networks (ANN) across three biologically-inspired neuron models. Built to develop intuition around training dynamics, neuron behaviour, and efficiency — not to produce publication-ready benchmarks.

> **Try the live dashboard →** deploy to Vercel or GitHub Pages — [see below](#deploy)

> [!IMPORTANT]
> **This is an experimental visualisation tool, not a rigorous benchmark.**
> - Default training curves are **synthetically generated** (seeded RNG), not measured runs
> - Energy is computed from real MAC counts via the **Horowitz 45 nm CMOS model** — not hardware-profiled
> - SNN curves during MNIST training are **derived by scaling ANN output**, not trained directly
> - Neuron dynamics are **simulated in isolation** — not embedded in a real spiking network

---

## Preview

<!-- After first deploy: take a screenshot, save to docs/preview.png, then uncomment the line below -->
<!-- ![Dashboard preview](docs/preview.png) -->

To add a screenshot:
1. Run `npm run dev`
2. Capture the full dashboard → save as `docs/preview.png`
3. For an animated GIF, record a short walkthrough → save as `docs/demo.gif`

---

## Interactive Demo

The dashboard is fully client-side — no server required. Visitors can click **▶ Train on MNIST** directly in the browser and watch real training curves stream in epoch by epoch.

### Option A — GitHub Pages

```bash
npm run build
# Push dist/ to the gh-pages branch, or configure GitHub Pages → Source: GitHub Actions
```

Add a shield badge once deployed:

```md
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://arturbarcij.github.io/snn-ann-dashboard)
```

### Option B — Vercel (one-click)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

Import the repo, set **Framework Preset → Vite**, deploy. Share the URL — anyone can open it and interact with the training dashboard instantly.

---

## Features

| Feature | Details | Data |
|---|---|---|
| **Epoch scrubber + playback** | Animate training convergence frame by frame | Synthetic by default; real after MNIST run |
| **Live MNIST training** | Trains a real ANN (MLP) via TensorFlow.js; SNN curves are derived approximations | ANN = real · SNN = scaled proxy |
| **3 neuron models** | LIF · Hodgkin-Huxley · Izhikevich tabs | Simulated — not trained networks |
| **Membrane potential traces** | V(t) waveforms showing spiking dynamics | Numerically simulated (not measured) |
| **Encoding comparison** | Rate coding vs temporal coding side by side | Synthetic spike trains |
| **Resource efficiency charts** | Power from Horowitz MAC/AC model; FLOPS = per-timestep ops; latency/memory synthetic | Power & FLOPS computed · rest estimated |
| **Cross-model summary table** | Final accuracy, energy gain, accuracy delta | Derived from synthetic or scaled data |

---

## Data Sources

> [!NOTE]
> **Summary of what is and isn't real in this dashboard:**
>
> | Data | Real? |
> |---|---|
> | ANN loss/accuracy (after clicking Train) | ✅ Real — TensorFlow.js trains on MNIST |
> | SNN loss/accuracy (after clicking Train) | ⚠️ Approximated — scaled from ANN output |
> | Default curves (before clicking Train) | ❌ Synthetic — seeded RNG, not training runs |
> | Energy / power figures | ⚠️ Computed — Horowitz 45 nm CMOS MAC/AC model, not hardware-profiled |
> | Membrane potential traces | ⚠️ Simulated — numerically integrated in isolation |
> | Spike rates per layer | ❌ Synthetic — literature-informed estimates |

### Synthetic baseline (default view)

> [!WARNING]
> The curves shown before clicking **▶ Train on MNIST** are **synthetically generated** and do not represent any real training run. They are seeded to look plausible but should not be cited or compared against real results.

Loss and accuracy curves follow:

$$\text{loss}(e) = 2.2 \cdot e^{-k \cdot e/N} + b + \varepsilon, \qquad \varepsilon \sim \mathcal{U}\!\left(-\tfrac{A}{2},\,\tfrac{A}{2}\right)$$

$$\text{acc}(e) = (\text{maxAcc} - 0.5)\left(1 - e^{-r \cdot e/N}\right) + \varepsilon$$

where $e$ is the epoch index, $N = 50$, and $k, r, b, A$ are per-model constants tuned to produce plausible divergence between ANN and SNN variants.

Spike rates per layer are **literature-informed estimates** — not measured from a trained network. Power and FLOPS are computed from these spike rates using the Horowitz energy model (see [Energy Calculation](#energy-calculation)). Latency and memory remain synthetic.

---

### MNIST training (experimental)

Clicking **▶ Train on MNIST** triggers real in-browser training for the ANN:

1. Downloads the MNIST dataset (~12 MB) via `fetch`
2. Trains a small CNN with **TensorFlow.js** for 15 epochs
3. Streams `{ loss, acc }` per epoch into the live charts

> [!WARNING]
> **SNN curves are not trained directly.** They are approximated by applying fixed scaling factors to the live ANN output. This is a deliberate simplification — actual SNN training (e.g. surrogate gradient descent with BPTT) is not yet implemented.

$$\text{loss}_\text{SNN} = \text{loss}_\text{ANN} \times s_\text{loss}$$
$$\text{acc}_\text{SNN} = \text{acc}_\text{ANN} \times s_\text{acc}$$

Scaling factors per neuron model (manually chosen to reflect literature-reported SNN accuracy gaps on MNIST — not fitted to real SNN runs):

| Model | Rate $s_\text{acc}$ | Temporal $s_\text{acc}$ | Rate $s_\text{loss}$ | Temporal $s_\text{loss}$ |
|---|---|---|---|---|
| LIF | 0.978 | 0.961 | 1.18 | 1.32 |
| Hodgkin-Huxley | 0.983 | 0.969 | 1.12 | 1.24 |
| Izhikevich | 0.980 | 0.964 | 1.15 | 1.28 |

Future work: replace scaled proxies with actual surrogate-gradient SNN training.

---

## Neuron Models

### Leaky Integrate-and-Fire (LIF)

The simplest biologically-inspired spiking neuron. Membrane potential leaks toward rest and fires when threshold is crossed.

$$\tau_m \frac{dV}{dt} = -(V - V_\text{rest}) + R \cdot I(t)$$

**Fire & reset:** $V \geq V_\text{thresh}$ → emit spike, $V \leftarrow V_\text{reset}$

| Parameter | Symbol | Value |
|---|---|---|
| Membrane time constant | $\tau_m$ | 20 ms |
| Resting potential | $V_\text{rest}$ | −70 mV |
| Firing threshold | $V_\text{thresh}$ | −55 mV |
| Reset potential | $V_\text{reset}$ | −70 mV |

Computationally cheap — widely used in neuromorphic hardware (Intel Loihi, IBM TrueNorth).

---

### Hodgkin-Huxley (H-H)

Biophysically detailed model with voltage-gated Na⁺ and K⁺ ion channels. Captures the full shape of biological action potentials at ~100× the compute cost of LIF.

$$C_m \frac{dV}{dt} = -g_{Na} m^3 h (V - E_{Na}) - g_K n^4 (V - E_K) - g_L (V - E_L) + I(t)$$

Gating variables $m$ (Na⁺ activation), $h$ (Na⁺ inactivation), $n$ (K⁺ activation) each follow:

$$\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)\, x$$

with rate functions:

$$\alpha_m = \frac{0.1(V+40)}{1 - e^{-(V+40)/10}}, \quad \beta_m = 4\, e^{-(V+65)/18}$$

$$\alpha_h = 0.07\, e^{-(V+65)/20}, \quad \beta_h = \frac{1}{1 + e^{-(V+35)/10}}$$

$$\alpha_n = \frac{0.01(V+55)}{1 - e^{-(V+55)/10}}, \quad \beta_n = 0.125\, e^{-(V+65)/80}$$

| Parameter | Value |
|---|---|
| $C_m$ (membrane capacitance) | 1 µF/cm² |
| $g_{Na}$ (Na⁺ max conductance) | 120 mS/cm² |
| $g_K$ (K⁺ max conductance) | 36 mS/cm² |
| $g_L$ (leak conductance) | 0.3 mS/cm² |
| $E_{Na}$ (Na⁺ reversal) | +50 mV |
| $E_K$ (K⁺ reversal) | −77 mV |
| $E_L$ (leak reversal) | −54.4 mV |

---

### Izhikevich

Efficient 2-variable model reproducing most known biological spiking patterns (regular, fast, bursting, chattering) at LIF-level compute cost.

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$$

$$\frac{du}{dt} = a(bv - u)$$

**Fire & reset:** $v \geq 30$ mV → $v \leftarrow c$, $u \leftarrow u + d$

| Parameter | Value | Role |
|---|---|---|
| $a$ | 0.02 | Recovery time scale |
| $b$ | 0.2 | Sensitivity of $u$ to sub-threshold $v$ |
| $c$ | −65 mV | After-spike reset of $v$ |
| $d$ | 8 | After-spike jump of recovery variable $u$ |

---

## Energy Calculation

> [!NOTE]
> Energy is computed from real MAC counts of the MNIST MLP and the **Horowitz (2014) 45 nm CMOS** energy-per-operation model. It is not profiled on hardware, but it is grounded in a published cost model rather than arbitrary numbers.

### Network architecture

The MLP trained in-browser (`mnist.js`) has the following weight layers:

| Layer | Shape | MACs |
|---|---|---|
| Dense-256 | 784 → 256 | 200,704 |
| Dense-128 | 256 → 128 | 32,768 |
| Dense-10  | 128 → 10  | 1,280 |
| **Total** | | **234,752** |

### ANN energy

Every weight fires on every forward pass (dense multiply-accumulate):

$$E_\text{ANN} = N_\text{MAC} \times E_\text{MAC} = 234{,}752 \times 4.6 \text{ pJ} \approx 1{,}080 \text{ nJ per inference}$$

### SNN energy

A weight only activates when its pre-synaptic neuron spikes (sparse accumulate). Summed over all weight layers and $T$ timesteps:

$$E_\text{SNN} = \sum_{l} N_{\text{MAC},l} \times r_{l-1} \times T \times E_\text{AC}$$

where $r_{l-1}$ is the spike rate of the layer feeding into layer $l$.

$$\text{Power (relative)} = \frac{E_\text{SNN}}{E_\text{ANN}} \times 100\%$$

$$\text{Energy Gain} = \frac{E_\text{ANN}}{E_\text{SNN}} = \frac{100}{\text{Power (relative)}}$$

### Constants

| Symbol | Value | Source |
|---|---|---|
| $E_\text{MAC}$ | 4.6 pJ | Horowitz, *1.1 Computing's Energy Problem*, ISSCC 2014 |
| $E_\text{AC}$ | 0.9 pJ | Horowitz 2014 (add only, no multiply) |
| $T$ | 10 timesteps | Typical rate-coded SNN inference window |
| $E_\text{MAC} / E_\text{AC}$ | ~5.1× | Operation-level advantage of spike events |

### Computed energy gains

Spike rates per layer are literature-informed estimates (not measured). The formula converts them to energy figures:

| Model | Coding | Input spike rate | Power (% of ANN) | Energy gain |
|---|---|---|---|---|
| LIF | Rate | 18% | 32.9% | **3.0×** |
| LIF | Temporal | 10% | 18.4% | **5.4×** |
| Hodgkin-Huxley | Rate | 20% | 36.8% | **2.7×** |
| Hodgkin-Huxley | Temporal | 10% | 18.7% | **5.4×** |
| Izhikevich | Rate | 16% | 29.6% | **3.4×** |
| Izhikevich | Temporal | 9% | 16.7% | **6.0×** |

**Key insight:** The input layer (Dense-256) accounts for 85% of all MACs (`200,704 / 234,752`), so the input spike rate dominates the energy estimate. Temporal coding achieves higher gains by encoding information in fewer spikes — at the cost of noise sensitivity.

FLOPS in the chart represent per-timestep SNN operations as a percentage of ANN MACs, showing the sparsity benefit independently of the $E_\text{MAC}/E_\text{AC}$ weighting.

---

## Encoding Schemes

### Rate coding

Spike probability is proportional to signal intensity:

$$P(\text{spike at } t) \propto x(t)$$

Higher information capacity, but requires many spikes — less efficient at inference time.

### Temporal coding (time-to-first-spike)

The **timing** of the first spike encodes signal magnitude. Stronger input → earlier spike:

$$t_\text{spike} \propto \frac{1}{x(t)}$$

Fewer spikes required — maximally efficient, but sensitive to temporal noise.

---

## Getting Started

```bash
git clone https://github.com/arturbarcij/snn-ann-dashboard
cd snn-ann-dashboard
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Click **▶ Train on MNIST** to replace synthetic curves with live in-browser training.

### Deploy

```bash
npm run build   # outputs to dist/
```

Host `dist/` on Vercel, Netlify, or GitHub Pages — no server needed.

---

## Stack

| Library | Role |
|---|---|
| React 18 + Vite | UI framework and dev server |
| Chart.js / react-chartjs-2 | Line, bar, and area charts |
| TensorFlow.js | In-browser MNIST MLP training (784 → 256 → 128 → 10) |
| Canvas API | Custom spike raster plots |

No external UI library. Styles are hand-written in `src/index.css`.

---

## Limitations & Planned Work

| Limitation | Status |
|---|---|
| SNN training curves are scaled ANN proxies | Planned: real surrogate-gradient SNN training |
| Default curves are synthetic RNG | Planned: replace with pre-trained checkpoint data |
| Energy figures are illustrative estimates | Planned: profile against real neuromorphic benchmarks |
| Neuron dynamics simulated in isolation | Out of scope for this dashboard |
| No weight visualisation or gradient flow | Out of scope for this dashboard |

---

## Project Context

Built as part of the thesis:
> *"Trade-offs in Performance and Computational Efficiency for Spiking Neural Networks"* — DTU, 2026

This dashboard is an **experimental companion tool** for developing intuition — not a standalone research artefact. Results and figures should not be cited without cross-referencing the thesis methodology.

Artur Barcij · [github.com/arturbarcij](https://github.com/arturbarcij)
