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
        if (S.view === "overlay" && img.finalOverlay) { overlayStyle = ""; overlaySrc = img.finalOverlay; }
        else if (S.view === "mask" && img.finalMask)  { overlayStyle = ""; overlaySrc = img.finalMask; }
        else if (S.view === "blackout" && img.finalBlackout) { overlayStyle = ""; overlaySrc = img.finalBlackout; }
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
            overlayImg.src = src;
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

// ── View toggle ────────────────────────────────────────────────────────────
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

document.getElementById("segmentAllBtn").addEventListener("click", runBatchSeg);
document.getElementById("textInput").addEventListener("keydown", e => { if (e.key === "Enter") runBatchSeg(); });
