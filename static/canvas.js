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

async function redrawWithOverlay(overlayUrl) {
    if (!overlayUrl) { currentOverlayImg = null; currentMaskImg = null; currentBlackoutImg = null; redraw(); return; }
    currentOverlayImg = await loadImgURL(overlayUrl);
    if (S.finalMask)     currentMaskImg     = await loadImgURL(S.finalMask);
    if (S.finalBlackout) currentBlackoutImg = await loadImgURL(S.finalBlackout);
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
    if (baseMask) { const bi = await loadImgURL(baseMask); cx.drawImage(bi, 0, 0); }
    const baseData  = cx.getImageData(0, 0, S.imgW, S.imgH);
    const paintData = paintCtx.getImageData(0, 0, S.imgW, S.imgH);
    const overlay = document.createElement("canvas"); overlay.width = S.imgW; overlay.height = S.imgH;
    const octx = overlay.getContext("2d"); const odata = octx.createImageData(S.imgW, S.imgH);
    const bcanvas = document.createElement("canvas"); bcanvas.width = S.imgW; bcanvas.height = S.imgH;
    const bctx = bcanvas.getContext("2d"); bctx.drawImage(baseImage, 0, 0);
    const bdata = bctx.getImageData(0, 0, S.imgW, S.imgH);
    const mcanvas = document.createElement("canvas"); mcanvas.width = S.imgW; mcanvas.height = S.imgH;
    const mctx = mcanvas.getContext("2d");
    mctx.fillStyle = "#000"; mctx.fillRect(0, 0, S.imgW, S.imgH);
    const mdata = mctx.getImageData(0, 0, S.imgW, S.imgH);
    let maskPx = 0;
    for (let i = 0; i < baseData.data.length; i += 4) {
        const baseMasked = baseData.data[i] > 127;
        const paintA = paintData.data[i+3];
        let isMasked = baseMasked;
        if (paintA > 10) isMasked = paintData.data[i] > 127;
        if (isMasked) {
            odata.data[i]=60; odata.data[i+1]=140; odata.data[i+2]=255; odata.data[i+3]=128;
            bdata.data[i]=0; bdata.data[i+1]=0; bdata.data[i+2]=0;
            mdata.data[i]=255; mdata.data[i+1]=255; mdata.data[i+2]=255;
            maskPx++;
        }
    }
    octx.putImageData(odata, 0, 0); bctx.putImageData(bdata, 0, 0); mctx.putImageData(mdata, 0, 0);
    const urls = await saveMaskToServer(S.images[S.activeIdx].id, overlay, mcanvas, bcanvas);
    S.finalOverlay = urls.overlay; S.finalMask = urls.mask; S.finalBlackout = urls.blackout;
    showScore(null, maskPx, S.imgW * S.imgH);
    await redrawWithOverlay(S.finalOverlay); updateUI();
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

// ── UI helpers ─────────────────────────────────────────────────────────────
function showScore(score, px, total) {
    document.getElementById("scoreSection").style.display = "block";
    document.getElementById("scoreVal").textContent = score ? score.toFixed(3) : "—";
    const pct = total ? ((px / total) * 100).toFixed(1) : "0";
    document.getElementById("scoreDet").textContent = `${px.toLocaleString()} / ${total.toLocaleString()} px (${pct}%)`;
}
function updateUI() {
    document.getElementById("downloadBtn").disabled = !S.finalOverlay;
    document.getElementById("invertMaskBtn").disabled = !S.finalMask;
    updatePolyUI();
}

async function invertCurrentMask() {
    if (S.activeIdx === null || !S.finalMask || !baseImage) {
        toast("No mask to invert yet");
        return;
    }

    showSpinner("Inverting mask...");
    try {
        const srcMask = await loadImgURL(S.finalMask);

        const maskCanvas = document.createElement("canvas");
        maskCanvas.width = S.imgW;
        maskCanvas.height = S.imgH;
        const mctx = maskCanvas.getContext("2d");
        mctx.drawImage(srcMask, 0, 0, S.imgW, S.imgH);

        const maskData = mctx.getImageData(0, 0, S.imgW, S.imgH);

        const overlayCanvas = document.createElement("canvas");
        overlayCanvas.width = S.imgW;
        overlayCanvas.height = S.imgH;
        const octx = overlayCanvas.getContext("2d");
        const odata = octx.createImageData(S.imgW, S.imgH);

        const blackoutCanvas = document.createElement("canvas");
        blackoutCanvas.width = S.imgW;
        blackoutCanvas.height = S.imgH;
        const bctx = blackoutCanvas.getContext("2d");
        bctx.drawImage(baseImage, 0, 0);
        const bdata = bctx.getImageData(0, 0, S.imgW, S.imgH);

        let maskPx = 0;
        for (let i = 0; i < maskData.data.length; i += 4) {
            const wasMasked = maskData.data[i] > 127;
            const isMasked = !wasMasked;

            if (isMasked) {
                maskData.data[i] = 255;
                maskData.data[i + 1] = 255;
                maskData.data[i + 2] = 255;
                maskData.data[i + 3] = 255;

                odata.data[i] = 60;
                odata.data[i + 1] = 140;
                odata.data[i + 2] = 255;
                odata.data[i + 3] = 128;

                bdata.data[i] = 0;
                bdata.data[i + 1] = 0;
                bdata.data[i + 2] = 0;
                maskPx++;
            } else {
                maskData.data[i] = 0;
                maskData.data[i + 1] = 0;
                maskData.data[i + 2] = 0;
                maskData.data[i + 3] = 255;
            }
        }

        mctx.putImageData(maskData, 0, 0);
        octx.putImageData(odata, 0, 0);
        bctx.putImageData(bdata, 0, 0);

        const urls = await saveMaskToServer(S.images[S.activeIdx].id, overlayCanvas, maskCanvas, blackoutCanvas);

        S.finalOverlay = urls.overlay;
        S.finalMask = urls.mask;
        S.finalBlackout = urls.blackout;

        // Inversion becomes the new baseline for subsequent polygon edits.
        S.prePaintOverlay = urls.overlay;
        S.prePaintMask = urls.mask;
        S.prePaintBlackout = urls.blackout;

        initPaintCanvas();
        showScore(null, maskPx, S.imgW * S.imgH);
        await redrawWithOverlay(S.finalOverlay);
        updateUI();

        const img = S.images[S.activeIdx];
        img.finalOverlay = S.finalOverlay;
        img.finalMask = S.finalMask;
        img.finalBlackout = S.finalBlackout;
        img.prePaintOverlay = S.prePaintOverlay;
        img.prePaintMask = S.prePaintMask;
        img.prePaintBlackout = S.prePaintBlackout;
        img.combinedPixels = maskPx;
        img.totalPixels = S.imgW * S.imgH;

        toast("Mask inverted");
    } catch (e) {
        toast("Mask inversion failed");
    } finally {
        hideSpinner();
    }
}

// ── Polygon mode buttons ───────────────────────────────────────────────────
document.querySelectorAll(".mode-btn").forEach(btn => btn.addEventListener("click", () => {
    S.polyMode = parseInt(btn.dataset.mode);
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
}));
document.getElementById("zoomReset").addEventListener("click", () => { resetZoom(); redrawCurrent(); });
document.getElementById("undoPolyBtn").addEventListener("click", () => undoLastPoly());
document.getElementById("invertMaskBtn").addEventListener("click", () => invertCurrentMask());
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

window.addEventListener("resize", () => { if (S.activeIdx !== null && baseImage) { fitCanvas(); if (S.finalOverlay) redrawWithOverlay(S.finalOverlay); else redraw(); } });
