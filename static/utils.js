// ── Utils ──────────────────────────────────────────────────────────────────
function loadImgURL(url) { return new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = url; }); }
function canvasToB64(c) { return c.toDataURL("image/png").replace("data:image/png;base64,", ""); }
function showSpinner(t) { document.getElementById("spinner").classList.add("on"); document.getElementById("spinnerText").textContent = t||"Processing…"; }
function hideSpinner() { document.getElementById("spinner").classList.remove("on"); }
function toast(m) { const t = document.getElementById("toast"); t.textContent = m; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 2800); }
function downloadDataUrl(url, name) { const a = document.createElement("a"); a.href = url; a.download = name; a.click(); }
async function downloadURL(url, name) {
    const blob = await fetch(url).then(r => r.blob());
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = objUrl; a.download = name; a.click();
    URL.revokeObjectURL(objUrl);
}
async function saveMaskToServer(imgId, overlayCanvas, maskCanvas, blackoutCanvas) {
    const r = await fetch("/save_mask", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            image_id: imgId,
            overlay_b64: canvasToB64(overlayCanvas),
            mask_b64: canvasToB64(maskCanvas),
            blackout_b64: canvasToB64(blackoutCanvas),
        }),
    });
    const d = await r.json();
    const t = Date.now();
    return { overlay: d.overlay + "?t=" + t, mask: d.mask + "?t=" + t, blackout: d.blackout + "?t=" + t };
}
