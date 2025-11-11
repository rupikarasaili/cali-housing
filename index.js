// ====== EDIT THESE to match your training setup ======
const MODEL_URL = 'diabetes_risk_model.onnx';
const PREDICTION_LABEL = 'Predicted value';

const FEATURE_NAMES = [
  // Put your feature names here in the SAME ORDER used to train (length = input_dim)
  // Example:
  // 'age','bmi','glucose','insulin','bp','skin','dpf','pregnancies'
];

const X_MEAN = [
  // Paste X_scaler.mean_.tolist() here
];

const X_STD = [
  // Paste X_scaler.scale_.tolist() here
];

const Y_MEAN = /* scalar */ 0;  // y_scaler.mean_[0]
const Y_STD  = /* scalar */ 1;  // y_scaler.scale_[0]
// =====================================================

const form = document.getElementById('form');
const inputsDiv = document.getElementById('inputs');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const errorEl = document.getElementById('error');
const predictBtn = document.getElementById('predictBtn');

function buildInputs() {
  inputsDiv.innerHTML = '';
  FEATURE_NAMES.forEach((name) => {
    const wrap = document.createElement('div');
    const label = document.createElement('label');
    label.textContent = name;
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.id = `f_${name}`;
    input.placeholder = name;
    wrap.appendChild(label);
    wrap.appendChild(input);
    inputsDiv.appendChild(wrap);
  });
}

function getFeatureVector() {
  const x = new Float32Array(FEATURE_NAMES.length);
  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    const v = Number(document.getElementById(`f_${FEATURE_NAMES[i]}`).value);
    if (Number.isNaN(v)) {
      throw new Error(`Missing/invalid value for "${FEATURE_NAMES[i]}"`);
    }
    x[i] = v;
  }
  return x;
}

function standardizeInPlace(x) {
  for (let i = 0; i < x.length; i++) {
    const mean = X_MEAN[i];
    const std  = X_STD[i] || 1e-8;
    x[i] = (x[i] - mean) / std;
  }
}

function inverseY(yScaled) {
  return yScaled * Y_STD + Y_MEAN;
}

let session = null;

async function loadModel() {
  try {
    statusEl.textContent = 'Model: loading…';
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    statusEl.textContent = 'Model: ready';
  } catch (e) {
    statusEl.textContent = 'Model: failed to load';
    errorEl.textContent = e.message || String(e);
    predictBtn.disabled = true;
  }
}

async function predict(e) {
  e.preventDefault();
  errorEl.textContent = '';
  resultEl.textContent = '';
  try {
    if (!session) throw new Error('Model not loaded');

    const x = getFeatureVector();          // raw
    standardizeInPlace(x);                 // StandardScaler(X)
    const inputTensor = new ort.Tensor('float32', x, [1, x.length]);

    const feeds = { input: inputTensor };  // input name = "input" (from export)
    const outputs = await session.run(feeds);
    const raw = outputs.output.data;       // output name = "output"
    const yScaled = raw[0];                // shape [1,1] -> take first
    const y = inverseY(yScaled);           // inverse-transform y

    resultEl.textContent = `${PREDICTION_LABEL}: ${y.toFixed(4)}`;
  } catch (err) {
    errorEl.textContent = err.message || String(err);
  }
}

buildInputs();
loadModel();
form.addEventListener('submit', predict);
