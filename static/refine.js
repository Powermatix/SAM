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

document.getElementById("backBtn").addEventListener("click", exitRefine);
