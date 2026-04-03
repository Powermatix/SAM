// ── Download ───────────────────────────────────────────────────────────────
document.getElementById("downloadBtn").addEventListener("click", async () => {
    if (S.activeIdx === null || !S.finalOverlay) return;
    const img = S.images[S.activeIdx];
    const baseName = img.name.replace(/\.[^.]+$/, "");
    if (S.view === "mask" && S.finalMask) {
        await downloadURL(S.finalMask, `${baseName}_mask.png`);
        toast("Downloaded mask");
    } else if (S.view === "blackout" && S.finalBlackout) {
        await downloadURL(S.finalBlackout, `${baseName}_blackout.png`);
        toast("Downloaded blackout");
    } else if (S.view === "original") {
        await downloadURL(`/uploads/${img.id}.png`, `${baseName}_original.png`);
        toast("Downloaded original");
    } else {
        // overlay view — composite base + overlay layer
        const c = document.createElement("canvas"); c.width = S.imgW; c.height = S.imgH;
        const cx = c.getContext("2d"); cx.drawImage(baseImage, 0, 0);
        const oi = await loadImgURL(S.finalOverlay); cx.drawImage(oi, 0, 0);
        downloadDataUrl(c.toDataURL("image/png"), `${baseName}_overlay.png`);
        toast("Downloaded overlay");
    }
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
    const base = await loadImgURL(`/uploads/${img.id}.png`);
    cx.drawImage(base, 0, 0);
    const ov = await loadImgURL(img.finalOverlay); cx.drawImage(ov, 0, 0);
    downloadDataUrl(c.toDataURL("image/png"), `${baseName}_overlay.png`);
    if (img.finalMask)     await downloadURL(img.finalMask,     `${baseName}_mask.png`);
    if (img.finalBlackout) await downloadURL(img.finalBlackout, `${baseName}_blackout.png`);
}
