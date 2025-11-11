const MODEL_URL = 'california_housing_model.onnx';
const PREDICTION_LABEL = 'Predicted value';

const FEATURE_NAMES = [
  'MedInc',
  'HouseAge',
  'AveRooms',
  'AveBedrms',
  'Population',
  'AveOccup',
  'Latitude',
  'Longitude'
];

const DEFAULT_ROW = [
  2.9063,      // MedInc
  27,          // HouseAge
  3.79927007,  // AveRooms
  1.11678832,  // AveBedrms
  2009,        // Population
  3.66605839,  // AveOccup
  33.88,       // Latitude
  -118.29      // Longitude
];
window.FEATURE_NAMES = FEATURE_NAMES;
window.DEFAULT_ROW  = DEFAULT_ROW;

const X_MEAN = [3.8724198816613273, 28.512631943242777, 5.458425832173263, 1.1015748607156226, 1416.1554767260773, 3.064019242740251, 35.645343470602306, -119.57148816342543] ;
const X_STD  = [1.9199429516327542, 12.577600124625755, 2.5283441973361325, 0.4687579558016747, 1093.299021908996, 7.636837328116874, 2.1427015636031084, 2.006692453629695] ;
const Y_MEAN = 1.0553631139156892 ;
const Y_STD  = 0.3564728269335749 ;


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
