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
