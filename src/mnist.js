import * as tf from "@tensorflow/tfjs";

// ─── MNIST data URLs (Google's TF.js tutorial CDN) ──────────
const IMAGES_URL = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const LABELS_URL  = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

const IMAGE_SIZE = 784;   // 28×28
const NUM_CLASSES = 10;
const TOTAL       = 65000;
const NUM_TRAIN   = 55000;
const NUM_TEST    = 10000;

// ─── Load MNIST ──────────────────────────────────────────────
// Downloads the sprite sheet + label file and returns raw Float32/Uint8 arrays.
export async function loadMnist(onStatus) {
  onStatus?.("Downloading MNIST images…");

  // Sprite sheet: 784px wide × 65000px tall, one image per row (grayscale stored in R channel)
  const imagesData = await new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "";
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx    = canvas.getContext("2d");
      const buf    = new ArrayBuffer(TOTAL * IMAGE_SIZE * 4); // Float32 bytes
      const chunkH = 5000;
      canvas.width  = img.naturalWidth;
      canvas.height = chunkH;

      for (let i = 0; i < TOTAL / chunkH; i++) {
        const view = new Float32Array(buf, i * IMAGE_SIZE * chunkH * 4, IMAGE_SIZE * chunkH);
        ctx.drawImage(img, 0, i * chunkH, img.naturalWidth, chunkH, 0, 0, img.naturalWidth, chunkH);
        const px = ctx.getImageData(0, 0, img.naturalWidth, chunkH).data;
        for (let j = 0; j < view.length; j++) view[j] = px[j * 4] / 255;
      }
      resolve(new Float32Array(buf));
    };
    img.onerror = reject;
    img.src = IMAGES_URL;
  });

  onStatus?.("Downloading MNIST labels…");
  const labelsResp = await fetch(LABELS_URL);
  const labelsData = new Uint8Array(await labelsResp.arrayBuffer());

  return { imagesData, labelsData };
}

// ─── ANN model ───────────────────────────────────────────────
// Simple MLP: 784 → 256 → 128 → 10
function buildModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [IMAGE_SIZE], units: 256, activation: "relu" }),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({ units: 128, activation: "relu" }),
      tf.layers.dense({ units: NUM_CLASSES, activation: "softmax" }),
    ],
  });
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

// ─── Train ───────────────────────────────────────────────────
// Trains the ANN on real MNIST data.
// Calls onEpochEnd(epochIndex, entry, fullHistory) after each epoch.
// Returns the full history array when complete.
export async function trainANN({ imagesData, labelsData, epochs = 15, onEpochEnd, signal }) {
  const trainXs = tf.tensor2d(imagesData.slice(0, NUM_TRAIN * IMAGE_SIZE), [NUM_TRAIN, IMAGE_SIZE]);
  const trainYs = tf.oneHot(
    tf.tensor1d(Array.from(labelsData.slice(0, NUM_TRAIN)), "int32"), NUM_CLASSES
  ).toFloat();
  const testXs = tf.tensor2d(imagesData.slice(NUM_TRAIN * IMAGE_SIZE), [NUM_TEST, IMAGE_SIZE]);
  const testYs = tf.oneHot(
    tf.tensor1d(Array.from(labelsData.slice(NUM_TRAIN)), "int32"), NUM_CLASSES
  ).toFloat();

  const model = buildModel();
  const history = [];

  await model.fit(trainXs, trainYs, {
    epochs,
    batchSize: 512,
    validationData: [testXs, testYs],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (signal?.aborted) { model.stopTraining = true; return; }
        const entry = {
          epoch: epoch + 1,
          ann: {
            loss: +logs.loss.toFixed(4),
            acc:  +(logs.val_acc * 100).toFixed(2),
          },
        };
        history.push(entry);
        onEpochEnd?.(epoch, entry, [...history]);
        await tf.nextFrame(); // yield to keep UI responsive
      },
    },
  });

  [trainXs, trainYs, testXs, testYs].forEach((t) => t.dispose());
  model.dispose();
  return history;
}
