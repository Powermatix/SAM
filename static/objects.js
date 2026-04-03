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
    showSpinner("Rebuilding mask…");
    const combined = document.createElement("canvas"); combined.width = S.imgW; combined.height = S.imgH;
    const cctx = combined.getContext("2d");
    cctx.fillStyle = "#000"; cctx.fillRect(0, 0, S.imgW, S.imgH);
    for (const obj of sel) { const img = await loadImgURL(obj.mask); cctx.drawImage(img, 0, 0); }
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
    const urls = await saveMaskToServer(S.images[S.activeIdx].id, overlay, combined, bcanvas);
    S.finalOverlay = urls.overlay; S.finalMask = urls.mask; S.finalBlackout = urls.blackout;
    S.prePaintOverlay = urls.overlay; S.prePaintMask = urls.mask; S.prePaintBlackout = urls.blackout;
    S.textCombinedMask = urls.mask;
    showScore(null, maskPx, S.imgW * S.imgH);
    hideSpinner();
    if (S.hasPaintEdits) await applyManualEdits(); else await redrawWithOverlay(S.finalOverlay);
    updateUI();
}
