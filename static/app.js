// ── State ──────────────────────────────────────────────────────────────────
const S = {
    images: [],        // array of image records
    activeIdx: null,   // null = batch view, number = refine mode
    view: "overlay",
    // Active-image working state
    imageId: null, imgW: 0, imgH: 0, scale: 1,
    textObjects: [],
    textCombinedOverlay: null, textCombinedMask: null, textCombinedBlackout: null,
    finalOverlay: null, finalMask: null, finalBlackout: null,
    prePaintOverlay: null, prePaintMask: null, prePaintBlackout: null,
    polyMode: 1,
    polyVerts: [], polyHistory: [], hasPaintEdits: false,
    zoom: 1, panX: 0, panY: 0,
    panning: false, panStart: null, spaceHeld: false,
};

const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");
let baseImage = null;
const OBJ_COLORS = ["#3c8cff","#ff6347","#32cd32","#ffd700","#da70d6","#00ced1","#ff69b4","#7b68ee"];
let paintCanvas = null, paintCtx = null;

function makeImgRecord(id, name, width, height) {
    return {
        id, name, width, height,
        status: "uploaded", errorMsg: null,
        textObjects: [],
        textCombinedOverlay: null, textCombinedMask: null, textCombinedBlackout: null,
        finalOverlay: null, finalMask: null, finalBlackout: null,
        prePaintOverlay: null, prePaintMask: null, prePaintBlackout: null,
        polyHistory: [], hasPaintEdits: false,
        paintData: null,  // saved ImageData when not active
        combinedPixels: 0, totalPixels: 0,
    };
}

// ── Paint canvas ───────────────────────────────────────────────────────────
function initPaintCanvas() {
    paintCanvas = document.createElement("canvas");
    paintCanvas.width = S.imgW; paintCanvas.height = S.imgH;
    paintCtx = paintCanvas.getContext("2d");
    paintCtx.clearRect(0, 0, S.imgW, S.imgH);
    S.hasPaintEdits = false;
    S.polyHistory = [];
    S.polyVerts = [];
    document.getElementById("clearAllBtn").disabled = true;
    document.getElementById("undoPolyBtn").disabled = true;
    document.getElementById("polyCount").textContent = "";
}

// ── Upload ─────────────────────────────────────────────────────────────────
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("dragging"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragging"));
uploadZone.addEventListener("drop", e => {
    e.preventDefault(); uploadZone.classList.remove("dragging");
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFiles(fileInput.files); fileInput.value = ""; });

async function handleFiles(fileList) {
    const files = [...fileList].filter(f => f.type.startsWith("image/"));
    if (!files.length) { toast("No image files found"); return; }
    uploadZone.classList.add("loading");
    toast(`Uploading ${files.length} image${files.length > 1 ? "s" : ""}…`);
    for (const file of files) {
        const fd = new FormData(); fd.append("image", file);
        try {
            const r = await fetch("/upload", { method: "POST", body: fd });
            const d = await r.json();
            if (d.error) { toast(`Failed: ${file.name}`); continue; }
            S.images.push(makeImgRecord(d.id, file.name, d.width, d.height));
        } catch(e) { toast(`Error: ${file.name}`); }
        renderBatchGrid(); renderImageList();
    }
    uploadZone.classList.remove("loading");
    document.getElementById("textInput").disabled = false;
    document.getElementById("segmentAllBtn").disabled = false;
    document.getElementById("imageListSection").style.display = "block";
    document.getElementById("placeholder").style.display = "none";
    document.getElementById("batchGrid").style.display = "grid";
    updateBatchUI();
    toast(`${S.images.length} image${S.images.length > 1 ? "s" : ""} ready — enter a text prompt`);
}

// ── Batch grid ─────────────────────────────────────────────────────────────
function renderBatchGrid() {
    document.getElementById("batchGrid").innerHTML =
        S.images.map((img, i) => makeCardHTML(img, i)).join("");
}

function makeCardHTML(img, idx) {
    const labels = { uploaded:"Ready", segmenting:"Segmenting…", done:"Done", error: img.errorMsg || "Error" };
    const badges = { uploaded:"badge-uploaded", segmenting:"badge-segmenting", done:"badge-done", error:"badge-error" };
    const isActive = S.activeIdx === idx;
    const isDone = img.status === "done";

    let overlayStyle = "display:none;", overlaySrc = "";
    if (isDone && img.finalOverlay) {
        if (S.view === "overlay" && img.finalOverlay) { overlayStyle = ""; overlaySrc = "data:image/png;base64," + img.finalOverlay; }
        else if (S.view === "mask" && img.finalMask)  { overlayStyle = ""; overlaySrc = "data:image/png;base64," + img.finalMask; }
        else if (S.view === "blackout" && img.finalBlackout) { overlayStyle = ""; overlaySrc = "data:image/png;base64," + img.finalBlackout; }
    }
    const baseStyle = (isDone && (S.view === "blackout" || S.view === "mask")) ? "display:none;" : "";
    const coverage = (isDone && img.totalPixels)
        ? ` · ${((img.combinedPixels / img.totalPixels) * 100).toFixed(1)}% covered` : "";

    return `<div class="img-card${isActive ? " active-card" : ""}" id="card-${idx}">
  <div class="card-thumb" onclick="openRefineOrToast(${idx})">
    <img class="card-base-img" src="/uploads/${img.id}.png" style="${baseStyle}" alt="">
    <img class="card-overlay-img" src="${overlaySrc}" style="${overlayStyle}" alt="">
    ${img.status === "segmenting" ? `<div class="card-spinner-wrap"><div class="spinner-ring"></div></div>` : ""}
    <div class="card-badge ${badges[img.status] || "badge-uploaded"}">${labels[img.status] || "Ready"}</div>
    ${isDone ? `<div class="card-refine-hint">✏ Refine</div>` : ""}
  </div>
  <div class="card-body">
    <div class="card-name" title="${img.name}">${img.name}</div>
    <div class="card-meta">${img.width}×${img.height}${coverage}</div>
    <div class="card-actions">
      <button class="btn btn-secondary btn-sm" onclick="event.stopPropagation();segmentOne(${idx})">↻ Segment</button>
      <button class="btn btn-primary btn-sm" onclick="event.stopPropagation();openRefineOrToast(${idx})"${!isDone ? " disabled" : ""}>✏ Refine</button>
    </div>
  </div>
</div>`;
}

function updateCard(idx) {
    const card = document.getElementById(`card-${idx}`);
    if (!card) return;
    const tmp = document.createElement("div");
    tmp.innerHTML = makeCardHTML(S.images[idx], idx);
    card.replaceWith(tmp.firstElementChild);
}

function updateBatchThumbnails() {
    S.images.forEach((img, idx) => {
        const card = document.getElementById(`card-${idx}`);
        if (!card || !img.finalOverlay) return;
        const baseImg = card.querySelector(".card-base-img");
        const overlayImg = card.querySelector(".card-overlay-img");
        if (!baseImg || !overlayImg) return;
        const srcMap = { overlay: img.finalOverlay, mask: img.finalMask, blackout: img.finalBlackout };
        const src = srcMap[S.view];
        if (src && S.view !== "original") {
            baseImg.style.display = (S.view === "blackout" || S.view === "mask") ? "none" : "";
            overlayImg.src = "data:image/png;base64," + src;
            overlayImg.style.display = "";
        } else {
            baseImg.style.display = "";
            overlayImg.style.display = "none";
        }
    });
}

function updateBatchUI() {
    const hasDone = S.images.some(img => img.status === "done");
    document.getElementById("downloadAllBtn").disabled = !hasDone;
    document.getElementById("imgCount").textContent = `(${S.images.length})`;
    if (S.images.length > 0) {
        document.getElementById("placeholder").style.display = "none";
        document.getElementById("batchGrid").style.display = "grid";
    }
}

// ── Image list (sidebar) ───────────────────────────────────────────────────
function renderImageList() {
    const dotClass = { uploaded:"sdot-uploaded", segmenting:"sdot-segmenting", done:"sdot-done", error:"sdot-error" };
    const statusTxt = { uploaded:"Ready", segmenting:"Segmenting…", done:"Done", error:"Error" };
    document.getElementById("imageList").innerHTML = S.images.map((img, i) => {
        const isActive = S.activeIdx === i;
        const clickable = img.status === "done";
        return `<div class="img-list-item${clickable ? " clickable" : ""}${isActive ? " active-item" : ""}"
                     ${clickable ? `onclick="openRefine(${i})"` : ""}>
            <img class="img-list-thumb" src="/uploads/${img.id}.png" alt="">
            <div class="img-list-info">
                <div class="img-list-name">${img.name}</div>
                <div class="img-list-sub">
                    <span class="sdot ${dotClass[img.status] || "sdot-uploaded"}"></span>
                    ${statusTxt[img.status] || "Ready"}
                </div>
            </div>
        </div>`;
    }).join("");
    updateBatchUI();
}

// ── Batch segmentation ─────────────────────────────────────────────────────
async function runBatchSeg() {
    const text = document.getElementById("textInput").value.trim();
    if (!text) { toast("Enter a text prompt first"); return; }
    if (!S.images.length) return;
    const btn = document.getElementById("segmentAllBtn");
    btn.disabled = true;
    document.getElementById("batchProgressDiv").style.display = "block";
    document.getElementById("batchTotal").textContent = S.images.length;
    for (let i = 0; i < S.images.length; i++) {
        document.getElementById("batchCur").textContent = i + 1;
        await segmentOne(i, text);
    }
    btn.disabled = false;
    document.getElementById("batchProgressDiv").style.display = "none";
    updateBatchUI();
    toast("Batch segmentation complete");
}

async function segmentOne(idx, text) {
    if (!text) text = document.getElementById("textInput").value.trim();
    if (!text) { toast("Enter a text prompt first"); return; }
    const img = S.images[idx];
    if (!img.id) return;
    img.status = "segmenting";
    updateCard(idx); renderImageList();
    try {
        const r = await fetch("/segment_text", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_id: img.id, text }),
        });
        const d = await r.json();
        if (d.found === 0) {
            img.status = "error"; img.errorMsg = "No objects found";
        } else {
            img.textObjects = d.objects.map(o => ({ ...o, selected: true }));
            img.textCombinedOverlay = d.combined_overlay;
            img.textCombinedMask    = d.combined_mask;
            img.textCombinedBlackout = d.combined_blackout;
            img.finalOverlay  = d.combined_overlay;
            img.finalMask     = d.combined_mask;
            img.finalBlackout = d.combined_blackout;
            img.prePaintOverlay  = d.combined_overlay;
            img.prePaintMask     = d.combined_mask;
            img.prePaintBlackout = d.combined_blackout;
            img.combinedPixels = d.combined_pixels;
            img.totalPixels    = d.total_pixels;
            img.status = "done"; img.errorMsg = null;
        }
    } catch(e) { img.status = "error"; img.errorMsg = "Request failed"; }
    updateCard(idx); renderImageList();
}

// ── Refine mode ────────────────────────────────────────────────────────────
function openRefineOrToast(idx) {
    if (S.images[idx].status !== "done") { toast("Segment this image first"); return; }
    openRefine(idx);
}

async function openRefine(idx) {
    if (S.images[idx].status !== "done") { toast("Segment this image first"); return; }
    if (S.activeIdx !== null) saveActiveState();
    S.activeIdx = idx;
    const img = S.images[idx];

    // Load state into working S fields
    S.imageId = img.id;
    S.imgW = img.width; S.imgH = img.height;
    S.textObjects = img.textObjects ? [...img.textObjects] : [];
    S.textCombinedOverlay = img.textCombinedOverlay;
    S.textCombinedMask    = img.textCombinedMask;
    S.textCombinedBlackout = img.textCombinedBlackout;
    S.finalOverlay  = img.finalOverlay;
    S.finalMask     = img.finalMask;
    S.finalBlackout = img.finalBlackout;
    S.prePaintOverlay  = img.prePaintOverlay;
    S.prePaintMask     = img.prePaintMask;
    S.prePaintBlackout = img.prePaintBlackout;
    S.polyVerts = [];

    // Switch UI
    document.getElementById("batchView").style.display = "none";
    document.getElementById("canvasArea").style.display = "flex";
    document.getElementById("batchSidebar").style.display = "none";
    document.getElementById("refineSidebar").style.display = "";

    // Fill refine sidebar info
    document.getElementById("refineImageName").textContent = img.name;
    document.getElementById("refineImageDims").textContent = `${img.width} × ${img.height} px`;
    document.getElementById("refineTextInput").value = document.getElementById("textInput").value;

    // Sync view buttons
    document.querySelectorAll(".view-btn").forEach(b => b.classList.toggle("active", b.dataset.view === S.view));

    // Load base image
    baseImage = await new Promise(r => {
        const i = new Image(); i.onload = () => r(i); i.src = `/uploads/${img.id}.png`;
    });

    // Init paint canvas then restore saved edits
    initPaintCanvas();
    if (img.paintData) {
        paintCtx.putImageData(img.paintData, 0, 0);
        S.hasPaintEdits = img.hasPaintEdits;
        S.polyHistory   = img.polyHistory ? [...img.polyHistory] : [];
        updatePolyUI();
    }

    fitCanvas(); resetZoom();
    if (S.finalOverlay) await redrawWithOverlay(S.finalOverlay); else redraw();

    // Sidebar sections
    if (S.textObjects.length > 0) renderObjects();
    else document.getElementById("objectsSection").style.display = "none";
    document.getElementById("polySection").style.display = "block";
    if (S.finalOverlay) {
        showScore(null, img.combinedPixels, img.totalPixels);
    } else {
        document.getElementById("scoreSection").style.display = "none";
    }
    updateUI();
    renderImageList();
}

function saveActiveState() {
    if (S.activeIdx === null) return;
    const img = S.images[S.activeIdx];
    img.textObjects          = S.textObjects ? [...S.textObjects] : [];
    img.textCombinedOverlay  = S.textCombinedOverlay;
    img.textCombinedMask     = S.textCombinedMask;
    img.textCombinedBlackout = S.textCombinedBlackout;
    img.finalOverlay  = S.finalOverlay;
    img.finalMask     = S.finalMask;
    img.finalBlackout = S.finalBlackout;
    img.prePaintOverlay  = S.prePaintOverlay;
    img.prePaintMask     = S.prePaintMask;
    img.prePaintBlackout = S.prePaintBlackout;
    img.polyHistory  = [...S.polyHistory];
    img.hasPaintEdits = S.hasPaintEdits;
    if (paintCanvas && S.hasPaintEdits) {
        img.paintData = paintCtx.getImageData(0, 0, S.imgW, S.imgH);
    } else {
        img.paintData = null;
    }
}

function exitRefine() {
    saveActiveState();
    S.activeIdx = null;
    S.polyVerts = [];
    currentOverlayImg = null; currentMaskImg = null; currentBlackoutImg = null;
    baseImage = null;
    document.getElementById("canvasArea").style.display = "none";
    document.getElementById("batchView").style.display = "flex";
    document.getElementById("refineSidebar").style.display = "none";
    document.getElementById("batchSidebar").style.display = "";
    document.getElementById("cursorHint").style.display = "none";
    updateZoomBadge();
    renderBatchGrid(); renderImageList(); updateBatchUI();
}

// ── Per-image segmentation (in refine mode) ───────────────────────────────
document.getElementById("refineSegBtn").addEventListener("click", runRefineSegment);
document.getElementById("refineTextInput").addEventListener("keydown", e => {
    if (e.key === "Enter") runRefineSegment();
});

async function runRefineSegment() {
    const text = document.getElementById("refineTextInput").value.trim();
    if (!text || S.activeIdx === null) return;
    const img = S.images[S.activeIdx];
    showSpinner("Segmenting…");
    try {
        const r = await fetch("/segment_text", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_id: img.id, text }),
        });
        const d = await r.json(); hideSpinner();
        if (d.found === 0) { toast("No objects found — try different text"); return; }
        S.textObjects = d.objects.map(o => ({ ...o, selected: true }));
        S.textCombinedOverlay = d.combined_overlay;
        S.textCombinedMask    = d.combined_mask;
        S.textCombinedBlackout = d.combined_blackout;
        S.finalOverlay  = d.combined_overlay;
        S.finalMask     = d.combined_mask;
        S.finalBlackout = d.combined_blackout;
        S.prePaintOverlay  = d.combined_overlay;
        S.prePaintMask     = d.combined_mask;
        S.prePaintBlackout = d.combined_blackout;
        initPaintCanvas();
        renderObjects();
        showScore(null, d.combined_pixels, d.total_pixels);
        toast(`Found ${d.found} object(s) — draw polygons to refine`);
        await redrawWithOverlay(S.finalOverlay);
        updateUI();
        // Sync back to image record
        img.textObjects      = [...S.textObjects];
        img.finalOverlay     = S.finalOverlay;
        img.finalMask        = S.finalMask;
        img.finalBlackout    = S.finalBlackout;
        img.prePaintOverlay  = S.prePaintOverlay;
        img.prePaintMask     = S.prePaintMask;
        img.prePaintBlackout = S.prePaintBlackout;
        img.combinedPixels   = d.combined_pixels;
        img.totalPixels      = d.total_pixels;
        img.status = "done";
    } catch(e) { hideSpinner(); toast("Segmentation failed"); }
}

// ── Objects panel ─────────────────────────────────────────────────────────
function renderObjects() {
    const sec = document.getElementById("objectsSection"); sec.style.display = "block";
    document.getElementById("objCount").textContent = `(${S.textObjects.length})`;
    document.getElementById("objectsList").innerHTML = S.textObjects.map((o, i) => `
        <div class="object-chip ${o.selected ? "selected" : ""}" onclick="toggleObj(${i})">
            <div class="obj-color" style="background:${OBJ_COLORS[i % OBJ_COLORS.length]}"></div>
            <div class="obj-info">Object ${i + 1}</div>
            <div class="obj-score">${o.score.toFixed(2)}</div>
        </div>`).join("");
}

function toggleObj(idx) { S.textObjects[idx].selected = !S.textObjects[idx].selected; renderObjects(); rebuildTextMask(); }

document.getElementById("selectAllBtn").addEventListener("click", () => {
    const allSel = S.textObjects.every(o => o.selected);
    S.textObjects.forEach(o => o.selected = !allSel);
    renderObjects(); rebuildTextMask();
});

async function rebuildTextMask() {
    const sel = S.textObjects.filter(o => o.selected);
    if (sel.length === 0) { S.finalOverlay = null; S.finalMask = null; S.finalBlackout = null; redraw(); updateUI(); return; }
    const combined = document.createElement("canvas"); combined.width = S.imgW; combined.height = S.imgH;
    const cctx = combined.getContext("2d");
    for (const obj of sel) { const img = await loadImgB64(obj.mask); cctx.drawImage(img, 0, 0); }
    const idata = cctx.getImageData(0, 0, S.imgW, S.imgH);
    const overlay = document.createElement("canvas"); overlay.width = S.imgW; overlay.height = S.imgH;
    const octx = overlay.getContext("2d"); const odata = octx.createImageData(S.imgW, S.imgH);
    const bcanvas = document.createElement("canvas"); bcanvas.width = S.imgW; bcanvas.height = S.imgH;
    const bctx = bcanvas.getContext("2d"); bctx.drawImage(baseImage, 0, 0);
    const bdata = bctx.getImageData(0, 0, S.imgW, S.imgH);
    let maskPx = 0;
    for (let i = 0; i < idata.data.length; i += 4) {
        if (idata.data[i] > 127) {
            odata.data[i]=60; odata.data[i+1]=140; odata.data[i+2]=255; odata.data[i+3]=128;
            bdata.data[i]=0; bdata.data[i+1]=0; bdata.data[i+2]=0; maskPx++;
        }
    }
    octx.putImageData(odata, 0, 0); bctx.putImageData(bdata, 0, 0);
    S.finalOverlay = canvasToB64(overlay); S.finalMask = canvasToB64(combined); S.finalBlackout = canvasToB64(bcanvas);
    S.prePaintOverlay = S.finalOverlay; S.prePaintMask = S.finalMask; S.prePaintBlackout = S.finalBlackout;
    S.textCombinedMask = S.finalMask;
    showScore(null, maskPx, S.imgW * S.imgH);
    if (S.hasPaintEdits) await applyManualEdits(); else redrawWithOverlay(S.finalOverlay);
    updateUI();
}

// ── Polygon interaction ────────────────────────────────────────────────────
function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: Math.round(((e.clientX - rect.left) - S.panX) / (S.scale * S.zoom)),
        y: Math.round(((e.clientY - rect.top) - S.panY) / (S.scale * S.zoom)),
    };
}

canvas.addEventListener("click", e => {
    if (S.activeIdx === null || S.panning || S.spaceHeld) return;
    e.preventDefault();
    const { x, y } = getCanvasCoords(e);
    if (S.polyVerts.length >= 3) {
        const first = S.polyVerts[0];
        if (Math.hypot(x - first.x, y - first.y) < 12 / (S.scale * S.zoom)) { closePolygon(); return; }
    }
    S.polyVerts.push({ x, y }); redrawCurrent();
});

canvas.addEventListener("dblclick", e => {
    if (S.activeIdx === null) return;
    e.preventDefault();
    if (S.polyVerts.length >= 3) closePolygon();
});

canvas.addEventListener("mousedown", e => {
    if (!baseImage) return;
    if (e.button === 1 || (S.spaceHeld && e.button === 0)) {
        e.preventDefault();
        S.panning = true;
        S.panStart = { x: e.clientX - S.panX, y: e.clientY - S.panY };
        canvas.style.cursor = "grabbing";
    }
});

canvas.addEventListener("mousemove", e => {
    if (S.panning && S.panStart) {
        S.panX = e.clientX - S.panStart.x;
        S.panY = e.clientY - S.panStart.y;
        redrawCurrent(); return;
    }
    const hint = document.getElementById("cursorHint");
    if (S.activeIdx !== null && !S.spaceHeld) {
        hint.style.display = "block";
        hint.style.left = e.clientX + "px"; hint.style.top = e.clientY + "px";
        hint.className = "cursor-hint " + (S.polyMode === 1 ? "pos" : "neg");
        hint.textContent = S.polyVerts.length === 0 ? (S.polyMode === 1 ? "Include" : "Exclude") : `${S.polyVerts.length} pts`;
    } else { hint.style.display = "none"; }
    if (S.polyVerts.length > 0 && !S.spaceHeld) {
        const { x, y } = getCanvasCoords(e);
        redrawCurrent();
        const es = S.scale * S.zoom;
        ctx.save(); ctx.translate(S.panX, S.panY); ctx.scale(es, es);
        const last = S.polyVerts[S.polyVerts.length - 1];
        const isPos = S.polyMode === 1;
        ctx.strokeStyle = isPos ? "rgba(34,197,94,0.5)" : "rgba(239,68,68,0.5)";
        ctx.lineWidth = 1.5 / es; ctx.setLineDash([6/es, 4/es]);
        ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(x, y); ctx.stroke();
        ctx.setLineDash([]);
        if (S.polyVerts.length >= 3) {
            const first = S.polyVerts[0];
            ctx.strokeStyle = isPos ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)";
            ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(first.x, first.y); ctx.stroke();
        }
        ctx.restore();
    }
});

canvas.addEventListener("mouseup", () => {
    if (S.panning) { S.panning = false; S.panStart = null; canvas.style.cursor = S.spaceHeld ? "grab" : "crosshair"; }
});
canvas.addEventListener("mouseleave", () => {
    document.getElementById("cursorHint").style.display = "none";
    if (S.panning) { S.panning = false; S.panStart = null; canvas.style.cursor = "crosshair"; }
});
canvas.addEventListener("contextmenu", e => { e.preventDefault(); cancelPolygon(); });

// ── Zoom ───────────────────────────────────────────────────────────────────
canvas.addEventListener("wheel", e => {
    if (!baseImage) return; e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const oldZoom = S.zoom;
    S.zoom = Math.max(0.5, Math.min(20, S.zoom * (e.deltaY < 0 ? 1.15 : 1/1.15)));
    const ratio = S.zoom / oldZoom;
    S.panX = mx - ratio * (mx - S.panX);
    S.panY = my - ratio * (my - S.panY);
    updateZoomBadge(); redrawCurrent();
}, { passive: false });

function cancelPolygon() { S.polyVerts = []; redrawCurrent(); }

async function closePolygon() {
    if (S.polyVerts.length < 3) { cancelPolygon(); return; }
    const mode = S.polyMode, verts = [...S.polyVerts];
    if (!paintCanvas) initPaintCanvas();
    paintCtx.globalCompositeOperation = "source-over";
    paintCtx.fillStyle = mode === 1 ? "white" : "black";
    paintCtx.beginPath(); paintCtx.moveTo(verts[0].x, verts[0].y);
    for (let i = 1; i < verts.length; i++) paintCtx.lineTo(verts[i].x, verts[i].y);
    paintCtx.closePath(); paintCtx.fill();
    S.hasPaintEdits = true;
    S.polyHistory.push({ verts, mode });
    S.polyVerts = [];
    showSpinner("Applying polygon…");
    await applyManualEdits();
    hideSpinner();
    updatePolyUI();
}

function updatePolyUI() {
    const n = S.polyHistory.length;
    document.getElementById("undoPolyBtn").disabled = n === 0;
    document.getElementById("clearAllBtn").disabled = n === 0 && !S.hasPaintEdits;
    document.getElementById("polyCount").textContent = n > 0 ? `${n} polygon${n > 1 ? "s" : ""} drawn` : "";
}

async function applyManualEdits() {
    if (!paintCanvas) return;
    const baseMask = S.prePaintMask || S.finalMask;
    if (!baseMask && !S.hasPaintEdits) return;
    const c = document.createElement("canvas"); c.width = S.imgW; c.height = S.imgH;
    const cx = c.getContext("2d");
    if (baseMask) { const bi = await loadImgB64(baseMask); cx.drawImage(bi, 0, 0); }
    const baseData  = cx.getImageData(0, 0, S.imgW, S.imgH);
    const paintData = paintCtx.getImageData(0, 0, S.imgW, S.imgH);
    const overlay = document.createElement("canvas"); overlay.width = S.imgW; overlay.height = S.imgH;
    const octx = overlay.getContext("2d"); const odata = octx.createImageData(S.imgW, S.imgH);
    const bcanvas = document.createElement("canvas"); bcanvas.width = S.imgW; bcanvas.height = S.imgH;
    const bctx = bcanvas.getContext("2d"); bctx.drawImage(baseImage, 0, 0);
    const bdata = bctx.getImageData(0, 0, S.imgW, S.imgH);
    const mcanvas = document.createElement("canvas"); mcanvas.width = S.imgW; mcanvas.height = S.imgH;
    const mctx = mcanvas.getContext("2d"); const mdata = mctx.createImageData(S.imgW, S.imgH);
    let maskPx = 0;
    for (let i = 0; i < baseData.data.length; i += 4) {
        const baseMasked = baseData.data[i] > 127;
        const paintA = paintData.data[i+3];
        let isMasked = baseMasked;
        if (paintA > 10) isMasked = paintData.data[i] > 127;
        if (isMasked) {
            odata.data[i]=60; odata.data[i+1]=140; odata.data[i+2]=255; odata.data[i+3]=128;
            bdata.data[i]=0; bdata.data[i+1]=0; bdata.data[i+2]=0;
            mdata.data[i]=255; mdata.data[i+1]=255; mdata.data[i+2]=255; mdata.data[i+3]=255;
            maskPx++;
        }
    }
    octx.putImageData(odata, 0, 0); bctx.putImageData(bdata, 0, 0); mctx.putImageData(mdata, 0, 0);
    S.finalOverlay = canvasToB64(overlay); S.finalMask = canvasToB64(mcanvas); S.finalBlackout = canvasToB64(bcanvas);
    showScore(null, maskPx, S.imgW * S.imgH);
    redrawWithOverlay(S.finalOverlay); updateUI();
    // Keep image record in sync so the card thumbnail is current after going back
    if (S.activeIdx !== null) {
        const img = S.images[S.activeIdx];
        img.finalOverlay = S.finalOverlay; img.finalMask = S.finalMask; img.finalBlackout = S.finalBlackout;
        img.combinedPixels = maskPx; img.totalPixels = S.imgW * S.imgH;
    }
}

async function undoLastPoly() {
    if (S.polyHistory.length === 0) return;
    showSpinner("Reverting…");
    S.polyHistory.pop();
    paintCtx.clearRect(0, 0, S.imgW, S.imgH);
    for (const p of S.polyHistory) {
        paintCtx.globalCompositeOperation = "source-over";
        paintCtx.fillStyle = p.mode === 1 ? "white" : "black";
        paintCtx.beginPath(); paintCtx.moveTo(p.verts[0].x, p.verts[0].y);
        for (let i = 1; i < p.verts.length; i++) paintCtx.lineTo(p.verts[i].x, p.verts[i].y);
        paintCtx.closePath(); paintCtx.fill();
    }
    S.hasPaintEdits = S.polyHistory.length > 0;
    if (S.hasPaintEdits) { await applyManualEdits(); }
    else {
        if (S.prePaintOverlay) {
            S.finalOverlay = S.prePaintOverlay; S.finalMask = S.prePaintMask; S.finalBlackout = S.prePaintBlackout;
            redrawWithOverlay(S.finalOverlay);
        } else { S.finalOverlay = null; S.finalMask = null; S.finalBlackout = null; redraw(); }
    }
    hideSpinner();
    updatePolyUI(); updateUI();
}

// ── Drawing ────────────────────────────────────────────────────────────────
let currentOverlayImg = null, currentMaskImg = null, currentBlackoutImg = null;

function applyZoomTransform() { ctx.translate(S.panX, S.panY); ctx.scale(S.scale * S.zoom, S.scale * S.zoom); }

function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save(); applyZoomTransform();
    if (baseImage) ctx.drawImage(baseImage, 0, 0);
    if (S.view === "overlay" && currentOverlayImg) ctx.drawImage(currentOverlayImg, 0, 0);
    drawPolyPreview(); ctx.restore();
}

function redrawCurrent() { if (S.finalOverlay) redrawWithOverlaySync(); else redraw(); }

function redrawWithOverlaySync() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save(); applyZoomTransform();
    if (S.view === "blackout" && S.finalBlackout && currentBlackoutImg) {
        ctx.drawImage(currentBlackoutImg, 0, 0);
    } else if (S.view === "mask" && S.finalMask && currentMaskImg) {
        ctx.fillStyle = "#000"; ctx.fillRect(0, 0, S.imgW, S.imgH);
        ctx.drawImage(currentMaskImg, 0, 0);
    } else {
        if (baseImage) ctx.drawImage(baseImage, 0, 0);
        if (S.view === "overlay" && currentOverlayImg) ctx.drawImage(currentOverlayImg, 0, 0);
    }
    drawPolyPreview(); ctx.restore();
}

async function redrawWithOverlay(overlayB64) {
    if (!overlayB64) { currentOverlayImg = null; currentMaskImg = null; currentBlackoutImg = null; redraw(); return; }
    currentOverlayImg = await loadImgB64(overlayB64);
    if (S.finalMask)     currentMaskImg     = await loadImgB64(S.finalMask);
    if (S.finalBlackout) currentBlackoutImg = await loadImgB64(S.finalBlackout);
    redrawWithOverlaySync();
}

function drawPolyPreview() {
    if (S.polyVerts.length === 0) return;
    const isPos = S.polyMode === 1;
    const color = isPos ? "#22c55e" : "#ef4444";
    const es = S.scale * S.zoom;
    if (S.polyVerts.length >= 3) {
        ctx.fillStyle = isPos ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)";
        ctx.beginPath(); ctx.moveTo(S.polyVerts[0].x, S.polyVerts[0].y);
        for (let i = 1; i < S.polyVerts.length; i++) ctx.lineTo(S.polyVerts[i].x, S.polyVerts[i].y);
        ctx.closePath(); ctx.fill();
    }
    ctx.strokeStyle = color; ctx.lineWidth = 2 / es;
    ctx.beginPath(); ctx.moveTo(S.polyVerts[0].x, S.polyVerts[0].y);
    for (let i = 1; i < S.polyVerts.length; i++) ctx.lineTo(S.polyVerts[i].x, S.polyVerts[i].y);
    ctx.stroke();
    const dotR = Math.max(3, 4 / es);
    for (let i = 0; i < S.polyVerts.length; i++) {
        ctx.beginPath(); ctx.arc(S.polyVerts[i].x, S.polyVerts[i].y, dotR, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? "white" : color; ctx.fill();
        ctx.strokeStyle = color; ctx.lineWidth = 1.5 / es; ctx.stroke();
    }
}

// ── Canvas fit & zoom ──────────────────────────────────────────────────────
function fitCanvas() {
    const a = document.getElementById("canvasArea");
    S.scale = Math.min((a.clientWidth - 40) / S.imgW, (a.clientHeight - 40) / S.imgH, 1);
    canvas.width  = Math.round(S.imgW * S.scale);
    canvas.height = Math.round(S.imgH * S.scale);
    resetZoom();
}
function resetZoom() { S.zoom = 1; S.panX = 0; S.panY = 0; updateZoomBadge(); }
function updateZoomBadge() {
    const badge = document.getElementById("zoomBadge");
    if (S.activeIdx === null) { badge.style.display = "none"; return; }
    badge.style.display = "flex";
    document.getElementById("zoomVal").textContent = Math.round(S.zoom * 100) + "%";
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function showScore(score, px, total) {
    document.getElementById("scoreSection").style.display = "block";
    document.getElementById("scoreVal").textContent = score ? score.toFixed(3) : "—";
    const pct = total ? ((px / total) * 100).toFixed(1) : "0";
    document.getElementById("scoreDet").textContent = `${px.toLocaleString()} / ${total.toLocaleString()} px (${pct}%)`;
}
function updateUI() {
    document.getElementById("downloadBtn").disabled = !S.finalOverlay;
    updatePolyUI();
}

// ── View toggle (shared across both sidebars) ──────────────────────────────
function setView(v) {
    S.view = v;
    document.querySelectorAll(".view-btn").forEach(b => b.classList.toggle("active", b.dataset.view === v));
    if (S.activeIdx !== null) {
        if (S.finalOverlay) redrawWithOverlay(S.finalOverlay); else redraw();
    } else {
        updateBatchThumbnails();
    }
}
document.querySelectorAll(".view-btn").forEach(btn => btn.addEventListener("click", () => setView(btn.dataset.view)));

// ── Buttons ────────────────────────────────────────────────────────────────
document.getElementById("backBtn").addEventListener("click", exitRefine);
document.getElementById("segmentAllBtn").addEventListener("click", runBatchSeg);
document.getElementById("textInput").addEventListener("keydown", e => { if (e.key === "Enter") runBatchSeg(); });
document.getElementById("undoPolyBtn").addEventListener("click", () => undoLastPoly());
document.getElementById("clearAllBtn").addEventListener("click", () => {
    initPaintCanvas();
    if (S.prePaintOverlay) {
        S.finalOverlay = S.prePaintOverlay; S.finalMask = S.prePaintMask; S.finalBlackout = S.prePaintBlackout;
        redrawWithOverlay(S.finalOverlay);
    } else {
        S.finalOverlay = null; S.finalMask = null; S.finalBlackout = null;
        document.getElementById("scoreSection").style.display = "none"; redraw();
    }
    updateUI(); toast("All manual edits cleared");
});
document.querySelectorAll(".mode-btn").forEach(btn => btn.addEventListener("click", () => {
    S.polyMode = parseInt(btn.dataset.mode);
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
}));
document.getElementById("zoomReset").addEventListener("click", () => { resetZoom(); redrawCurrent(); });

// ── Download ───────────────────────────────────────────────────────────────
document.getElementById("downloadBtn").addEventListener("click", async () => {
    if (S.activeIdx === null || !S.finalOverlay) return;
    const img = S.images[S.activeIdx];
    const baseName = img.name.replace(/\.[^.]+$/, "");
    const c = document.createElement("canvas"); c.width = S.imgW; c.height = S.imgH;
    const cx = c.getContext("2d"); cx.drawImage(baseImage, 0, 0);
    const oi = await loadImgB64(S.finalOverlay); cx.drawImage(oi, 0, 0);
    downloadDataUrl(c.toDataURL("image/png"), `${baseName}_overlay.png`);
    if (S.finalMask)     downloadDataUrl("data:image/png;base64," + S.finalMask,     `${baseName}_mask.png`);
    if (S.finalBlackout) downloadDataUrl("data:image/png;base64," + S.finalBlackout, `${baseName}_blackout.png`);
    toast("Downloaded overlay, mask & blackout");
});

document.getElementById("downloadAllBtn").addEventListener("click", async () => {
    const done = S.images.filter(img => img.finalOverlay);
    if (!done.length) { toast("No segmented images to download"); return; }
    for (const img of done) {
        await downloadImageResults(img);
        await new Promise(r => setTimeout(r, 120));
    }
    toast(`Downloaded ${done.length} image${done.length > 1 ? "s" : ""}`);
});

async function downloadImageResults(img) {
    const baseName = img.name.replace(/\.[^.]+$/, "");
    const c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
    const cx = c.getContext("2d");
    const base = await new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = `/uploads/${img.id}.png`; });
    cx.drawImage(base, 0, 0);
    const ov = await loadImgB64(img.finalOverlay); cx.drawImage(ov, 0, 0);
    downloadDataUrl(c.toDataURL("image/png"), `${baseName}_overlay.png`);
    if (img.finalMask)     downloadDataUrl("data:image/png;base64," + img.finalMask,     `${baseName}_mask.png`);
    if (img.finalBlackout) downloadDataUrl("data:image/png;base64," + img.finalBlackout, `${baseName}_blackout.png`);
}

// ── Keyboard ───────────────────────────────────────────────────────────────
document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT") return;
    if (S.activeIdx === null) return;
    if (e.key === "1") { S.polyMode = 1; document.querySelectorAll(".mode-btn").forEach(b=>b.classList.remove("active")); document.getElementById("modePos").classList.add("active"); }
    if (e.key === "2") { S.polyMode = 0; document.querySelectorAll(".mode-btn").forEach(b=>b.classList.remove("active")); document.getElementById("modeNeg").classList.add("active"); }
    if (e.key === "Escape") cancelPolygon();
    if ((e.ctrlKey||e.metaKey) && e.key === "z") { e.preventDefault(); undoLastPoly(); }
    if (e.key === "0") { resetZoom(); redrawCurrent(); }
    if (e.key === " ") { e.preventDefault(); S.spaceHeld = true; canvas.style.cursor = "grab"; }
});
document.addEventListener("keyup", e => {
    if (e.key === " ") { S.spaceHeld = false; if (!S.panning) canvas.style.cursor = "crosshair"; }
});

// ── Utils ──────────────────────────────────────────────────────────────────
function loadImgB64(b64) { return new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = "data:image/png;base64," + b64; }); }
function canvasToB64(c) { return c.toDataURL("image/png").replace("data:image/png;base64,", ""); }
function showSpinner(t) { document.getElementById("spinner").classList.add("on"); document.getElementById("spinnerText").textContent = t||"Processing…"; }
function hideSpinner() { document.getElementById("spinner").classList.remove("on"); }
function toast(m) { const t = document.getElementById("toast"); t.textContent = m; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 2800); }
function downloadDataUrl(url, name) { const a = document.createElement("a"); a.href = url; a.download = name; a.click(); }
window.addEventListener("resize", () => { if (S.activeIdx !== null && baseImage) { fitCanvas(); if (S.finalOverlay) redrawWithOverlay(S.finalOverlay); else redraw(); } });
