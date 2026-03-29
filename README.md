# SNN vs ANN · Training Dashboard

An interactive React dashboard comparing Spiking Neural Networks (SNN) and Artificial Neural Networks (ANN) trained on MNIST — with mock data, epoch scrubbing, and live metric cards.

## Stack

- React 18 + Vite
- Chart.js via react-chartjs-2
- Zero external UI dependencies

## Getting started

```bash
npm install
npm run dev
```

## Features

- **Epoch scrubber** — drag or hit Play to animate training curves
- **Training loss** — SNN vs ANN convergence comparison
- **Validation accuracy** — tracks model improvement over epochs
- **Spike rate by layer** — realistic LIF neuron firing distribution
- **Resource usage** — power, FLOPS, and latency: SNN vs ANN

## Background

Spiking Neural Networks process information using sparse binary spikes, mimicking biological neurons. This dashboard visualises the core trade-off: SNNs achieve ~3.86× energy reduction at a ~1.5% accuracy cost on MNIST, with spike rates as low as 5% in output layers.

Built as part of the thesis: *"Trade-offs in Performance and Computational Efficiency for Spiking Neural Networks"* — DTU , 2026.
