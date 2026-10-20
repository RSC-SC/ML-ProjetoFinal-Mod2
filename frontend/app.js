/**
 * Frontend de demonstração do ScanVision.
 *
 * Envia a imagem para POST /detect (multipart/form-data) e desenha as
 * bounding boxes do JSON recebido sobre um <canvas>. A API em si continua
 * devolvendo apenas JSON — nada de imagem desenhada no backend.
 */

"use strict";

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const endpointInput = document.getElementById("endpoint");
const statusEl = document.getElementById("status");
const imagePanel = document.getElementById("image-panel");
const resultsPanel = document.getElementById("results-panel");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const rowsEl = document.getElementById("rows");
const countEl = document.getElementById("count");
const jsonEl = document.getElementById("json");

let currentImage = null; // HTMLImageElement carregado

// ---------- interação ----------

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

["dragover", "dragenter"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("hover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("hover");
  })
);
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// ---------- fluxo principal ----------

async function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    setStatus("error", `Arquivo inválido: envie uma imagem (o arquivo é '${file.type || "desconhecido"}').`);
    return;
  }

  try {
    await loadImage(file);
  } catch {
    setStatus("error", "Não foi possível decodificar a imagem no navegador.");
    return;
  }

  imagePanel.hidden = false;
  resultsPanel.hidden = true;
  drawImage();

  const form = new FormData();
  form.append("file", file);

  setStatus("info", "Enviando para a API...");
  const t0 = performance.now();
  try {
    const response = await fetch(endpointInput.value.trim(), { method: "POST", body: form });
    const elapsed = performance.now() - t0;
    const data = await response.json();

    if (!response.ok) {
      setStatus("error", `API respondeu ${response.status}: ${data.detail ?? "erro desconhecido"}`);
      return;
    }

    renderDetections(data);
    const n = data.detections.length;
    setStatus(
      "ok",
      `${n} detecção(ões) · ${elapsed.toFixed(0)} ms de ida e volta (rede + API)`
    );
  } catch (err) {
    setStatus("error", `Falha ao chamar a API: ${err.message}. O uvicorn está rodando?`);
  }
}

// ---------- desenho ----------

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      currentImage = img;
      resolve();
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

/** Desenha a imagem em tamanho nativo; o CSS apenas reduz a exibição. */
function drawImage() {
  canvas.width = currentImage.naturalWidth;
  canvas.height = currentImage.naturalHeight;
  ctx.drawImage(currentImage, 0, 0);
}

/**
 * Desenha cada caixa do JSON na coordenada original (pixels da imagem).
 * Cores são determinísticas por classe (hash do nome -> matiz).
 */
function drawBoxes(detections) {
  drawImage();
  detections.forEach((det) => {
    const { x_min, y_min, x_max, y_max } = det.box;
    const color = classColor(det.class_name);

    ctx.lineWidth = Math.max(2, Math.round(canvas.width / 400));
    ctx.strokeStyle = color;
    ctx.strokeRect(x_min, y_min, x_max - x_min, y_max - y_min);

    // Rótulo "classe 92%" acima da caixa (ou dentro se perto do topo).
    const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
    const fontSize = Math.max(12, Math.round(canvas.width / 60));
    ctx.font = `${fontSize}px system-ui, sans-serif`;
    const textWidth = ctx.measureText(label).width;
    const pad = 4;
    const boxHeight = fontSize + pad * 2;
    const labelY = y_min - boxHeight >= 0 ? y_min - boxHeight : y_min;

    ctx.fillStyle = color;
    ctx.fillRect(x_min, labelY, textWidth + pad * 2, boxHeight);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x_min + pad, labelY + boxHeight - pad - fontSize / 3);
  });
}

/** Matiz estável por nome de classe, sem depender da ordem das detecções. */
function classColor(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) % 360;
  return `hsl(${hash}, 85%, 45%)`;
}

// ---------- resultados ----------

function renderDetections(data) {
  drawBoxes(data.detections);

  countEl.textContent = `${data.detections.length}`;
  rowsEl.innerHTML = "";
  data.detections.forEach((det) => {
    const tr = document.createElement("tr");

    const tdClass = document.createElement("td");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = classColor(det.class_name);
    tdClass.appendChild(swatch);
    tdClass.appendChild(document.createTextNode(det.class_name));

    const tdConf = document.createElement("td");
    tdConf.textContent = `${(det.confidence * 100).toFixed(1)}%`;

    const tdBox = document.createElement("td");
    tdBox.className = "mono";
    const b = det.box;
    tdBox.textContent = `${b.x_min}, ${b.y_min}, ${b.x_max}, ${b.y_max}`;

    tr.append(tdClass, tdConf, tdBox);
    rowsEl.appendChild(tr);
  });

  jsonEl.textContent = JSON.stringify(data, null, 2);
  resultsPanel.hidden = false;
}

function setStatus(kind, message) {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = message;
}
