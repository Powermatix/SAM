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
