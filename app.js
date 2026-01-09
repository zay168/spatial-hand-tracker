/**
 * Spatial Hand Tracker - visionOS Experience
 * Apple Vision Pro inspired interactions with MediaPipe
 */

import { HandLandmarker, FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest';

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
    mediapipe: {
        numHands: 2,
        minDetectionConfidence: 0.7,
        minPresenceConfidence: 0.7,
        minTrackingConfidence: 0.7,
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5
    },

    interaction: {
        pinchThreshold: 0.065,
        pinchReleaseThreshold: 0.085,
        grabRadius: 90,
        smoothing: 0.3,
        gestureConfirmFrames: 3
    },

    rendering: {
        skeleton: {
            jointRadius: 4,
            boneWidth: 2.5,
            colors: ['#ff9500', '#5ac8fa', '#bf5af2', '#ff375f', '#30d158', '#ffffff']
        },
        uiUpdateInterval: 80
    }
};

// Hand bone connections with color indices
const HAND_CONNECTIONS = [
    [0, 1, 0], [1, 2, 0], [2, 3, 0], [3, 4, 0],
    [0, 5, 1], [5, 6, 1], [6, 7, 1], [7, 8, 1],
    [0, 9, 2], [9, 10, 2], [10, 11, 2], [11, 12, 2],
    [0, 13, 3], [13, 14, 3], [14, 15, 3], [15, 16, 3],
    [0, 17, 4], [17, 18, 4], [18, 19, 4], [19, 20, 4],
    [5, 9, 5], [9, 13, 5], [13, 17, 5]
];

// ============================================
// FILTERS FOR PRECISION
// ============================================
class OneEuroFilter {
    constructor(minCutoff = 1.0, beta = 0.007) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.xPrev = null;
        this.dxPrev = 0;
        this.tPrev = null;
    }

    filter(x, t) {
        if (this.xPrev === null) {
            this.xPrev = x;
            this.tPrev = t;
            return x;
        }

        const dt = Math.max((t - this.tPrev) / 1000, 0.001);
        this.tPrev = t;

        const dx = (x - this.xPrev) / dt;
        const alpha = this.alpha(1.0, dt);
        const edx = alpha * dx + (1 - alpha) * this.dxPrev;
        this.dxPrev = edx;

        const cutoff = this.minCutoff + this.beta * Math.abs(edx);
        const result = this.alpha(cutoff, dt) * x + (1 - this.alpha(cutoff, dt)) * this.xPrev;
        this.xPrev = result;

        return result;
    }

    alpha(cutoff, dt) {
        const tau = 1.0 / (2 * Math.PI * cutoff);
        return 1.0 / (1.0 + tau / dt);
    }

    reset() {
        this.xPrev = null;
        this.dxPrev = 0;
        this.tPrev = null;
    }
}

class GestureStabilizer {
    constructor(frames = 3) {
        this.requiredFrames = frames;
        this.currentState = false;
        this.pendingState = false;
        this.frameCount = 0;
    }

    update(raw) {
        if (raw !== this.pendingState) {
            this.pendingState = raw;
            this.frameCount = 1;
        } else {
            this.frameCount++;
        }

        if (this.frameCount >= this.requiredFrames && this.pendingState !== this.currentState) {
            this.currentState = this.pendingState;
        }

        return this.currentState;
    }

    reset() {
        this.currentState = false;
        this.pendingState = false;
        this.frameCount = 0;
    }
}

// ============================================
// STATE
// ============================================
const state = {
    handLandmarker: null,
    faceLandmarker: null,
    isRunning: false,

    fps: 0,
    fpsCount: 0,
    lastFpsTime: 0,
    lastUiUpdate: 0,

    canvas: null,
    ctx: null,
    canvasW: 0,
    canvasH: 0,

    // Filters for each hand (support 2 hands)
    handFilters: [
        { filterX: new OneEuroFilter(1.0, 0.007), filterY: new OneEuroFilter(1.0, 0.007), pinchStabilizer: new GestureStabilizer(CONFIG.interaction.gestureConfirmFrames) },
        { filterX: new OneEuroFilter(1.0, 0.007), filterY: new OneEuroFilter(1.0, 0.007), pinchStabilizer: new GestureStabilizer(CONFIG.interaction.gestureConfirmFrames) }
    ],

    // Legacy single hand support (using first hand)
    filterX: new OneEuroFilter(1.0, 0.007),
    filterY: new OneEuroFilter(1.0, 0.007),
    pinchStabilizer: new GestureStabilizer(CONFIG.interaction.gestureConfirmFrames),

    cursor: { x: 0, y: 0 },
    cursors: [{ x: 0, y: 0 }, { x: 0, y: 0 }], // Multi-cursor support
    isPinching: false,
    pinchDistance: 0,
    handSize: 0.15,
    handsData: [], // Store data for all detected hands

    // Middle finger detection (doigt d'honneur)
    middleFingerDetected: false,
    middleFingerHand: null,
    middleFingerZoom: 0, // 0 to 1 zoom progress
    middleFingerStabilizer: new GestureStabilizer(5),

    // Face tracking
    faceDetected: false,
    faceLandmarks: null,
    faceBox: null,

    objects: [],
    grabbedObject: null,
    grabOffset: { x: 0, y: 0 },
    nearestObject: null,

    dropZone: { x: 0, y: 0, w: 0, h: 0 },
    isOverDropZone: false,
    itemsInZone: 0
};

// ============================================
// DOM ELEMENTS
// ============================================
const $ = id => document.getElementById(id);
const el = {
    webcam: $('webcam'),
    canvas: $('skeletonCanvas'),
    cameraWindow: $('cameraWindow'),
    spatialObjects: $('spatialObjects'),

    permissionScreen: $('permissionScreen'),
    loadingScreen: $('loadingScreen'),
    loadingText: $('loadingText'),
    startButton: $('startButton'),
    resetButton: $('resetButton'),

    handPointer: $('handPointer'),
    dropZone: $('dropZone'),
    dropCount: $('dropCount'),

    fpsPill: $('fpsPill'),
    handsPill: $('handsPill'),
    gesturePill: $('gesturePill'),

    objectsGrid: $('objectsGrid'),
    cursorData: $('cursorData'),
    pinchData: $('pinchData'),
    holdingData: $('holdingData'),

    notifGrabbed: $('notifGrabbed'),
    notifDropped: $('notifDropped'),
    notifMiddleFinger: $('notifMiddleFinger'),

    // Hand pointers for multi-hand
    handPointer2: $('handPointer2'),

    // Middle finger overlay
    middleFingerOverlay: $('middleFingerOverlay'),
    middleFingerText: $('middleFingerText'),

    // Face indicator
    faceIndicator: $('faceIndicator')
};

// ============================================
// INITIALIZATION
// ============================================
async function init() {
    try {
        el.loadingScreen.classList.remove('hidden');
        el.loadingText.textContent = 'Loading AI models...';

        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );

        el.loadingText.textContent = 'Initializing hand tracker...';

        state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numHands: CONFIG.mediapipe.numHands,
            minHandDetectionConfidence: CONFIG.mediapipe.minDetectionConfidence,
            minHandPresenceConfidence: CONFIG.mediapipe.minPresenceConfidence,
            minTrackingConfidence: CONFIG.mediapipe.minTrackingConfidence
        });

        el.loadingText.textContent = 'Initializing face tracker...';

        state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numFaces: CONFIG.mediapipe.numFaces,
            minFaceDetectionConfidence: CONFIG.mediapipe.minFaceDetectionConfidence,
            minFacePresenceConfidence: CONFIG.mediapipe.minFacePresenceConfidence
        });

        setupCanvas();
        initObjects();
        initDropZone();

        el.loadingScreen.classList.add('hidden');

        el.startButton.onclick = startCamera;
        el.resetButton.onclick = resetObjects;
        window.onresize = handleResize;

    } catch (err) {
        console.error('Init error:', err);
        el.loadingText.textContent = `Error: ${err.message}`;
    }
}

function setupCanvas() {
    state.canvas = el.canvas;
    state.ctx = state.canvas.getContext('2d', { alpha: true, desynchronized: true });
}

function initObjects() {
    const objectEls = el.spatialObjects.querySelectorAll('.floating-object');

    objectEls.forEach(objEl => {
        state.objects.push({
            id: objEl.id,
            el: objEl,
            x: parseFloat(objEl.style.left) || 15,
            y: parseFloat(objEl.style.top) || 30,
            origX: parseFloat(objEl.style.left) || 15,
            origY: parseFloat(objEl.style.top) || 30,
            type: objEl.dataset.type,
            isGrabbed: false,
            inZone: false
        });
    });
}

function initDropZone() {
    updateDropZone();
    window.addEventListener('resize', updateDropZone);
}

function updateDropZone() {
    const rect = el.dropZone.getBoundingClientRect();
    const containerRect = el.cameraWindow.getBoundingClientRect();

    state.dropZone = {
        x: ((rect.left - containerRect.left + rect.width / 2) / containerRect.width) * 100,
        y: ((rect.top - containerRect.top + rect.height / 2) / containerRect.height) * 100,
        w: (rect.width / containerRect.width) * 100,
        h: (rect.height / containerRect.height) * 100
    };
}

function handleResize() {
    const container = el.cameraWindow;
    state.canvas.width = container.clientWidth;
    state.canvas.height = container.clientHeight;
    state.canvasW = container.clientWidth;
    state.canvasH = container.clientHeight;
    updateDropZone();
}

// ============================================
// CAMERA
// ============================================
async function startCamera() {
    try {
        el.permissionScreen.classList.add('hidden');
        el.loadingScreen.classList.remove('hidden');
        el.loadingText.textContent = 'Accessing camera...';

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user',
                frameRate: { ideal: 30 }
            }
        });

        el.webcam.srcObject = stream;
        await el.webcam.play();

        handleResize();
        el.loadingScreen.classList.add('hidden');

        state.isRunning = true;
        requestAnimationFrame(processFrame);

    } catch (err) {
        console.error('Camera error:', err);
        el.loadingText.textContent = `Camera error: ${err.message}`;
        el.permissionScreen.classList.remove('hidden');
    }
}

// ============================================
// FRAME PROCESSING
// ============================================
function processFrame(timestamp) {
    if (!state.isRunning) return;

    // FPS
    state.fpsCount++;
    if (timestamp - state.lastFpsTime >= 1000) {
        state.fps = state.fpsCount;
        state.fpsCount = 0;
        state.lastFpsTime = timestamp;
    }

    // Hand Detection
    const handResults = state.handLandmarker.detectForVideo(el.webcam, timestamp);

    // Face Detection
    const faceResults = state.faceLandmarker.detectForVideo(el.webcam, timestamp);

    // Clear
    state.ctx.clearRect(0, 0, state.canvasW, state.canvasH);

    const numHands = handResults.landmarks?.length || 0;
    const numFaces = faceResults.faceLandmarks?.length || 0;

    // Process faces
    state.faceDetected = numFaces > 0;
    if (numFaces > 0) {
        state.faceLandmarks = faceResults.faceLandmarks[0];
        renderFace(state.faceLandmarks);
    }
    updateFaceIndicator();

    // Reset middle finger detection
    let middleFingerFound = false;
    let middleFingerHandLandmarks = null;

    if (numHands > 0) {
        // Process all detected hands
        state.ctx.save();
        state.ctx.translate(state.canvasW, 0);
        state.ctx.scale(-1, 1);

        for (let i = 0; i < numHands; i++) {
            const landmarks = handResults.landmarks[i];
            const handedness = handResults.handednesses?.[i]?.[0]?.categoryName || 'Unknown';

            // Render skeleton for each hand with different colors
            renderSkeleton(landmarks, i);

            // Check for middle finger gesture on each hand
            if (detectMiddleFinger(landmarks)) {
                middleFingerFound = true;
                middleFingerHandLandmarks = landmarks;
            }
        }

        state.ctx.restore();

        // Process interaction with first hand for object grabbing
        const landmarks = handResults.landmarks[0];
        processInteraction(landmarks, timestamp);

        // Update second hand pointer if present
        if (numHands > 1 && el.handPointer2) {
            const landmarks2 = handResults.landmarks[1];
            updateSecondPointer(landmarks2, timestamp);
        } else if (el.handPointer2) {
            el.handPointer2.classList.remove('visible');
        }

    } else {
        el.handPointer.classList.remove('visible', 'pinching', 'near-object');
        if (el.handPointer2) el.handPointer2.classList.remove('visible');

        if (state.grabbedObject) {
            releaseObject();
        }

        resetFilters();
    }

    // Handle middle finger detection with stabilization
    const stableMiddleFinger = state.middleFingerStabilizer.update(middleFingerFound);

    if (stableMiddleFinger && middleFingerHandLandmarks) {
        state.middleFingerDetected = true;
        state.middleFingerHand = middleFingerHandLandmarks;
        // Smooth zoom in
        state.middleFingerZoom = Math.min(1, state.middleFingerZoom + 0.08);
    } else {
        state.middleFingerDetected = false;
        // Smooth zoom out
        state.middleFingerZoom = Math.max(0, state.middleFingerZoom - 0.05);
    }

    updateMiddleFingerOverlay();

    // UI update (throttled)
    if (timestamp - state.lastUiUpdate >= CONFIG.rendering.uiUpdateInterval) {
        updateUI(numHands, numFaces);
        state.lastUiUpdate = timestamp;
    }

    requestAnimationFrame(processFrame);
}

function resetFilters() {
    state.filterX.reset();
    state.filterY.reset();
    state.pinchStabilizer.reset();
}

// ============================================
// SKELETON RENDERING (Elegant Version)
// ============================================
function renderSkeleton(landmarks, handIndex = 0) {
    const ctx = state.ctx;
    const w = state.canvasW;
    const h = state.canvasH;

    // Elegant color palettes with gradients
    const palettes = [
        {
            primary: '#5ac8fa',
            secondary: '#bf5af2',
            accent: '#ff9500',
            glow: 'rgba(90, 200, 250, 0.4)'
        },
        {
            primary: '#00ff88',
            secondary: '#ff6b9d',
            accent: '#ffbe0b',
            glow: 'rgba(0, 255, 136, 0.4)'
        }
    ];
    const palette = palettes[handIndex % 2];

    // Pre-calculate all positions
    const positions = new Array(21);
    for (let i = 0; i < 21; i++) {
        positions[i] = {
            x: landmarks[i].x * w,
            y: landmarks[i].y * h,
            z: landmarks[i].z || 0
        };
    }

    // Draw glow layer first (background)
    ctx.save();
    ctx.shadowColor = palette.glow;
    ctx.shadowBlur = 15;
    ctx.strokeStyle = palette.glow;
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Draw all connections as glow
    ctx.beginPath();
    for (const [start, end] of HAND_CONNECTIONS) {
        ctx.moveTo(positions[start].x, positions[start].y);
        ctx.lineTo(positions[end].x, positions[end].y);
    }
    ctx.stroke();
    ctx.restore();

    // Finger groups for gradient coloring
    const fingerGroups = [
        { indices: [[0, 1], [1, 2], [2, 3], [3, 4]], name: 'thumb' },
        { indices: [[0, 5], [5, 6], [6, 7], [7, 8]], name: 'index' },
        { indices: [[0, 9], [9, 10], [10, 11], [11, 12]], name: 'middle' },
        { indices: [[0, 13], [13, 14], [14, 15], [15, 16]], name: 'ring' },
        { indices: [[0, 17], [17, 18], [18, 19], [19, 20]], name: 'pinky' }
    ];

    // Draw bones with gradient effect
    fingerGroups.forEach((finger, fingerIdx) => {
        finger.indices.forEach(([start, end], segIdx) => {
            const p1 = positions[start];
            const p2 = positions[end];

            // Create gradient along the bone
            const gradient = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
            const hue = (fingerIdx * 50 + handIndex * 180) % 360;
            gradient.addColorStop(0, `hsla(${hue}, 80%, 65%, 0.9)`);
            gradient.addColorStop(1, `hsla(${(hue + 30) % 360}, 70%, 55%, 0.9)`);

            ctx.beginPath();
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 3 - segIdx * 0.3; // Taper towards fingertips
            ctx.lineCap = 'round';
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        });
    });

    // Draw palm connections
    const palmConnections = [[5, 9], [9, 13], [13, 17]];
    ctx.beginPath();
    ctx.strokeStyle = `hsla(${handIndex * 180}, 60%, 50%, 0.7)`;
    ctx.lineWidth = 2;
    palmConnections.forEach(([start, end]) => {
        ctx.moveTo(positions[start].x, positions[start].y);
        ctx.lineTo(positions[end].x, positions[end].y);
    });
    ctx.stroke();

    // Draw elegant joints with depth effect
    const jointSizes = [8, 5, 4, 3, 4]; // Wrist, MCP, PIP, DIP, Tips

    for (let i = 0; i < 21; i++) {
        const pos = positions[i];
        const fingerIdx = i === 0 ? 0 : Math.floor((i - 1) / 4);
        const jointType = i === 0 ? 0 : ((i - 1) % 4) + 1;
        const size = jointSizes[jointType] || 4;

        // Depth-based size adjustment
        const depthScale = 1 + (pos.z * 2);
        const radius = size * Math.max(0.5, Math.min(1.5, depthScale));

        const hue = (fingerIdx * 50 + handIndex * 180) % 360;

        // Outer glow
        const glowGradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, radius * 2);
        glowGradient.addColorStop(0, `hsla(${hue}, 80%, 70%, 0.8)`);
        glowGradient.addColorStop(0.5, `hsla(${hue}, 70%, 60%, 0.3)`);
        glowGradient.addColorStop(1, 'transparent');

        ctx.beginPath();
        ctx.fillStyle = glowGradient;
        ctx.arc(pos.x, pos.y, radius * 2, 0, Math.PI * 2);
        ctx.fill();

        // Core joint
        const coreGradient = ctx.createRadialGradient(
            pos.x - radius * 0.3, pos.y - radius * 0.3, 0,
            pos.x, pos.y, radius
        );
        coreGradient.addColorStop(0, '#ffffff');
        coreGradient.addColorStop(0.3, `hsla(${hue}, 80%, 75%, 1)`);
        coreGradient.addColorStop(1, `hsla(${hue}, 70%, 50%, 1)`);

        ctx.beginPath();
        ctx.fillStyle = coreGradient;
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fill();

        // Highlight
        ctx.beginPath();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.arc(pos.x - radius * 0.3, pos.y - radius * 0.3, radius * 0.3, 0, Math.PI * 2);
        ctx.fill();
    }
}

// ============================================
// FACE MESH TRIANGLES (Complete coverage - ~200 triangles)
// ============================================
const FACE_MESH_TRIANGLES = [
    // === FOREHEAD (dense coverage) ===
    [10, 338, 297], [338, 297, 299], [297, 299, 332], [299, 332, 333],
    [332, 333, 284], [333, 284, 298], [284, 298, 251], [298, 251, 301],
    [251, 301, 389], [301, 389, 368], [389, 368, 356], [368, 356, 264],
    [356, 264, 454], [264, 454, 447], [454, 447, 323], [447, 323, 366],
    [323, 366, 361], [366, 361, 401], [361, 401, 288], [401, 288, 435],
    [10, 109, 67], [109, 67, 103], [67, 103, 54], [103, 54, 21],
    [54, 21, 162], [21, 162, 127], [162, 127, 234], [127, 234, 93],
    [10, 338, 151], [338, 151, 108], [151, 108, 69], [108, 69, 104],
    [69, 104, 68], [104, 68, 71], [68, 71, 139], [71, 139, 70],

    // === LEFT EYE REGION ===
    [33, 246, 161], [246, 161, 160], [161, 160, 159], [160, 159, 158],
    [159, 158, 157], [158, 157, 173], [157, 173, 133], [173, 133, 155],
    [133, 155, 154], [155, 154, 153], [154, 153, 145], [153, 145, 144],
    [145, 144, 163], [144, 163, 7], [163, 7, 33], [7, 33, 246],
    [246, 33, 130], [33, 130, 25], [130, 25, 110], [25, 110, 24],
    [110, 24, 23], [24, 23, 22], [23, 22, 26], [22, 26, 112],
    [26, 112, 226], [112, 226, 31], [226, 31, 228], [31, 228, 229],
    [228, 229, 230], [229, 230, 231], [230, 231, 232], [231, 232, 233],

    // === RIGHT EYE REGION ===
    [263, 466, 388], [466, 388, 387], [388, 387, 386], [387, 386, 385],
    [386, 385, 384], [385, 384, 398], [384, 398, 362], [398, 362, 382],
    [362, 382, 381], [382, 381, 380], [381, 380, 374], [380, 374, 373],
    [374, 373, 390], [373, 390, 249], [390, 249, 263], [249, 263, 466],
    [466, 263, 359], [263, 359, 255], [359, 255, 339], [255, 339, 254],
    [339, 254, 253], [254, 253, 252], [253, 252, 256], [252, 256, 341],
    [256, 341, 446], [341, 446, 261], [446, 261, 448], [261, 448, 449],
    [448, 449, 450], [449, 450, 451], [450, 451, 452], [451, 452, 453],

    // === NOSE ===
    [168, 6, 197], [6, 197, 195], [197, 195, 5], [195, 5, 4],
    [5, 4, 1], [4, 1, 19], [1, 19, 94], [19, 94, 2],
    [94, 2, 164], [2, 164, 0], [164, 0, 11], [0, 11, 12],
    [11, 12, 13], [12, 13, 14], [13, 14, 15], [14, 15, 16],
    [168, 417, 6], [417, 6, 419], [6, 419, 197], [419, 197, 248],
    [197, 248, 195], [248, 195, 456], [195, 456, 5], [456, 5, 420],
    [98, 97, 2], [97, 2, 326], [2, 326, 327], [326, 327, 278],
    [327, 278, 168], [278, 168, 6], [98, 64, 97], [64, 97, 75],

    // === LEFT CHEEK ===
    [234, 227, 137], [227, 137, 177], [137, 177, 215], [177, 215, 138],
    [215, 138, 135], [138, 135, 169], [135, 169, 170], [169, 170, 140],
    [170, 140, 171], [140, 171, 175], [171, 175, 152], [175, 152, 148],
    [234, 93, 132], [93, 132, 58], [132, 58, 172], [58, 172, 136],
    [172, 136, 150], [136, 150, 149], [150, 149, 176], [149, 176, 148],
    [116, 117, 118], [117, 118, 119], [118, 119, 120], [119, 120, 121],
    [120, 121, 47], [121, 47, 126], [47, 126, 100], [126, 100, 101],

    // === RIGHT CHEEK ===
    [454, 447, 366], [447, 366, 401], [366, 401, 435], [401, 435, 367],
    [435, 367, 364], [367, 364, 394], [364, 394, 395], [394, 395, 369],
    [395, 369, 396], [369, 396, 400], [396, 400, 377], [400, 377, 152],
    [454, 323, 361], [323, 361, 288], [361, 288, 397], [288, 397, 365],
    [397, 365, 379], [365, 379, 378], [379, 378, 400], [378, 400, 377],
    [345, 346, 347], [346, 347, 348], [347, 348, 349], [348, 349, 350],
    [349, 350, 277], [350, 277, 355], [277, 355, 329], [355, 329, 330],

    // === UPPER LIP ===
    [61, 185, 40], [185, 40, 39], [40, 39, 37], [39, 37, 0],
    [37, 0, 267], [0, 267, 269], [267, 269, 270], [269, 270, 409],
    [270, 409, 291], [61, 146, 91], [146, 91, 181], [91, 181, 84],
    [78, 191, 80], [191, 80, 81], [80, 81, 82], [81, 82, 13],
    [82, 13, 312], [13, 312, 311], [312, 311, 310], [311, 310, 415],

    // === LOWER LIP ===
    [78, 95, 88], [95, 88, 178], [88, 178, 87], [178, 87, 14],
    [87, 14, 317], [14, 317, 402], [317, 402, 318], [402, 318, 324],
    [318, 324, 308], [181, 84, 17], [84, 17, 314], [17, 314, 405],
    [314, 405, 321], [405, 321, 375], [321, 375, 291], [310, 415, 308],

    // === CHIN ===
    [152, 148, 176], [148, 176, 149], [176, 149, 150], [149, 150, 136],
    [150, 136, 172], [136, 172, 58], [172, 58, 132], [58, 132, 93],
    [152, 377, 400], [377, 400, 378], [400, 378, 379], [378, 379, 365],
    [379, 365, 397], [365, 397, 288], [397, 288, 361], [288, 361, 323],
    [175, 171, 152], [171, 152, 396], [152, 396, 377], [396, 377, 369],

    // === JAW LINE LEFT ===
    [132, 123, 50], [123, 50, 187], [50, 187, 207], [187, 207, 216],
    [207, 216, 212], [216, 212, 202], [212, 202, 204], [202, 204, 194],
    [204, 194, 201], [194, 201, 200], [201, 200, 199], [200, 199, 175],

    // === JAW LINE RIGHT ===
    [361, 352, 280], [352, 280, 411], [280, 411, 427], [411, 427, 436],
    [427, 436, 432], [436, 432, 422], [432, 422, 424], [422, 424, 418],
    [424, 418, 421], [418, 421, 420], [421, 420, 419], [420, 419, 400],

    // === BETWEEN EYES ===
    [9, 107, 55], [107, 55, 65], [55, 65, 52], [65, 52, 53],
    [52, 53, 46], [53, 46, 124], [46, 124, 35], [124, 35, 111],
    [9, 336, 285], [336, 285, 295], [285, 295, 282], [295, 282, 283],
    [282, 283, 276], [283, 276, 353], [276, 353, 265], [353, 265, 340],

    // === LEFT EYEBROW ===
    [70, 63, 105], [63, 105, 66], [105, 66, 107], [66, 107, 55],
    [107, 55, 65], [55, 65, 52], [65, 52, 53], [52, 53, 46],

    // === RIGHT EYEBROW ===
    [300, 293, 334], [293, 334, 296], [334, 296, 336], [296, 336, 285],
    [336, 285, 295], [285, 295, 282], [295, 282, 283], [282, 283, 276],

    // === TEMPLE LEFT ===
    [127, 234, 162], [234, 162, 21], [162, 21, 54], [21, 54, 103],
    [54, 103, 67], [103, 67, 109], [67, 109, 10], [109, 10, 151],

    // === TEMPLE RIGHT ===
    [356, 454, 389], [454, 389, 251], [389, 251, 284], [251, 284, 332],
    [284, 332, 297], [332, 297, 338], [297, 338, 10], [338, 10, 151],

    // === Additional fill triangles ===
    [234, 127, 93], [127, 93, 132], [93, 132, 234], [454, 356, 323],
    [356, 323, 361], [323, 361, 454], [1, 4, 2], [4, 2, 98],
    [2, 98, 1], [61, 146, 185], [146, 185, 61], [291, 375, 409],
    [375, 409, 291], [152, 175, 377], [175, 377, 152]
];


// ============================================
// FACE RENDERING (Optimized Triangle Mesh)
// ============================================
function renderFace(landmarks) {
    const ctx = state.ctx;
    const w = state.canvasW;
    const h = state.canvasH;

    if (landmarks.length < 400) return; // Need full mesh

    ctx.save();
    ctx.translate(state.canvasW, 0);
    ctx.scale(-1, 1);

    // Pre-calculate positions for optimization
    const positions = new Float32Array(landmarks.length * 2);
    for (let i = 0; i < landmarks.length; i++) {
        positions[i * 2] = landmarks[i].x * w;
        positions[i * 2 + 1] = landmarks[i].y * h;
    }

    // Draw triangle mesh
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = 'rgba(90, 200, 250, 0.25)';
    ctx.fillStyle = 'rgba(90, 200, 250, 0.03)';

    // Batch draw triangles
    ctx.beginPath();
    for (let i = 0; i < FACE_MESH_TRIANGLES.length; i++) {
        const [a, b, c] = FACE_MESH_TRIANGLES[i];

        // Bounds check
        if (a >= landmarks.length || b >= landmarks.length || c >= landmarks.length) continue;

        const ax = positions[a * 2], ay = positions[a * 2 + 1];
        const bx = positions[b * 2], by = positions[b * 2 + 1];
        const cx = positions[c * 2], cy = positions[c * 2 + 1];

        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.lineTo(cx, cy);
        ctx.lineTo(ax, ay);
    }
    ctx.stroke();
    ctx.fill();

    // Draw enhanced features
    drawFaceFeatures(ctx, positions, w, h, landmarks);

    ctx.restore();
}

// ============================================
// FACE FEATURES (Eyes, Lips, Eyebrows)
// ============================================
function drawFaceFeatures(ctx, positions, w, h, landmarks) {
    // Left eye contour
    const leftEyeUpper = [246, 161, 160, 159, 158, 157, 173, 133];
    const leftEyeLower = [33, 7, 163, 144, 145, 153, 154, 155, 133];

    // Right eye contour
    const rightEyeUpper = [466, 388, 387, 386, 385, 384, 398, 362];
    const rightEyeLower = [263, 249, 390, 373, 374, 380, 381, 382, 362];

    // Lips
    const outerLipsUpper = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291];
    const outerLipsLower = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61];
    const innerLipsUpper = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];
    const innerLipsLower = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];

    // Eyebrows
    const leftEyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
    const rightEyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];

    // Draw eyes with gradient
    ctx.save();

    // Eye glow effect
    ctx.shadowColor = 'rgba(90, 200, 250, 0.5)';
    ctx.shadowBlur = 8;

    // Left eye
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(90, 200, 250, 0.8)';
    ctx.lineWidth = 1.5;
    drawContour(ctx, positions, leftEyeUpper);
    drawContour(ctx, positions, leftEyeLower);
    ctx.stroke();

    // Right eye
    ctx.beginPath();
    drawContour(ctx, positions, rightEyeUpper);
    drawContour(ctx, positions, rightEyeLower);
    ctx.stroke();

    // Draw iris points
    const leftIris = [468, 469, 470, 471, 472];
    const rightIris = [473, 474, 475, 476, 477];

    ctx.fillStyle = 'rgba(90, 200, 250, 0.9)';
    [leftIris, rightIris].forEach(iris => {
        if (iris[0] < landmarks.length) {
            const cx = positions[iris[0] * 2];
            const cy = positions[iris[0] * 2 + 1];

            const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, 8);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
            gradient.addColorStop(0.3, 'rgba(90, 200, 250, 0.6)');
            gradient.addColorStop(1, 'rgba(90, 200, 250, 0.1)');

            ctx.beginPath();
            ctx.fillStyle = gradient;
            ctx.arc(cx, cy, 8, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    ctx.restore();

    // Lips with gradient
    ctx.save();
    ctx.shadowColor = 'rgba(255, 100, 130, 0.4)';
    ctx.shadowBlur = 6;

    const lipGradient = ctx.createLinearGradient(
        positions[61 * 2], positions[61 * 2 + 1],
        positions[291 * 2], positions[291 * 2 + 1]
    );
    lipGradient.addColorStop(0, 'rgba(255, 120, 150, 0.6)');
    lipGradient.addColorStop(0.5, 'rgba(255, 80, 120, 0.7)');
    lipGradient.addColorStop(1, 'rgba(255, 120, 150, 0.6)');

    ctx.strokeStyle = lipGradient;
    ctx.lineWidth = 2;

    ctx.beginPath();
    drawContour(ctx, positions, outerLipsUpper, true);
    ctx.stroke();

    ctx.beginPath();
    drawContour(ctx, positions, outerLipsLower, true);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(255, 100, 130, 0.4)';
    ctx.lineWidth = 1;

    ctx.beginPath();
    drawContour(ctx, positions, innerLipsUpper, true);
    ctx.stroke();

    ctx.beginPath();
    drawContour(ctx, positions, innerLipsLower, true);
    ctx.stroke();

    ctx.restore();

    // Eyebrows
    ctx.save();
    ctx.strokeStyle = 'rgba(180, 160, 140, 0.6)';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';

    ctx.beginPath();
    drawContour(ctx, positions, leftEyebrow);
    ctx.stroke();

    ctx.beginPath();
    drawContour(ctx, positions, rightEyebrow);
    ctx.stroke();

    ctx.restore();

    // Face contour glow
    const faceOval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];

    ctx.save();
    ctx.strokeStyle = 'rgba(90, 200, 250, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 6]);

    ctx.beginPath();
    drawContour(ctx, positions, faceOval, true);
    ctx.stroke();

    ctx.restore();
}

// Helper function to draw smooth contours
function drawContour(ctx, positions, indices, closePath = false) {
    if (indices.length === 0) return;

    const points = indices.map(i => ({
        x: positions[i * 2],
        y: positions[i * 2 + 1]
    }));

    ctx.moveTo(points[0].x, points[0].y);

    // Use quadratic curves for smoothness
    for (let i = 1; i < points.length - 1; i++) {
        const xc = (points[i].x + points[i + 1].x) / 2;
        const yc = (points[i].y + points[i + 1].y) / 2;
        ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
    }

    // Last point
    ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);

    if (closePath) {
        ctx.closePath();
    }
}

// ============================================
// MIDDLE FINGER DETECTION
// ============================================
function detectMiddleFinger(landmarks) {
    // Landmark indices:
    // Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
    // Tips: 4, 8, 12, 16, 20
    // MCP (knuckles): 1, 5, 9, 13, 17

    const wrist = landmarks[0];
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const middleTip = landmarks[12];
    const ringTip = landmarks[16];
    const pinkyTip = landmarks[20];

    const thumbMcp = landmarks[2];
    const indexMcp = landmarks[5];
    const middleMcp = landmarks[9];
    const ringMcp = landmarks[13];
    const pinkyMcp = landmarks[17];

    // Check if middle finger is extended (tip is far from wrist relative to MCP)
    const middleExtended = distance2D(middleTip, wrist) > distance2D(middleMcp, wrist) * 1.5;

    // Check if other fingers are curled (tips close to palm)
    const thumbCurled = distance2D(thumbTip, wrist) < distance2D(thumbMcp, wrist) * 1.3;
    const indexCurled = distance2D(indexTip, wrist) < distance2D(indexMcp, wrist) * 1.4;
    const ringCurled = distance2D(ringTip, wrist) < distance2D(ringMcp, wrist) * 1.4;
    const pinkyCurled = distance2D(pinkyTip, wrist) < distance2D(pinkyMcp, wrist) * 1.4;

    // Middle finger gesture: middle extended, others curled
    const isMiddleFingerGesture = middleExtended && indexCurled && ringCurled && pinkyCurled;

    return isMiddleFingerGesture;
}

function distance2D(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

// ============================================
// MIDDLE FINGER OVERLAY
// ============================================
function updateMiddleFingerOverlay() {
    if (!el.middleFingerOverlay) return;

    if (state.middleFingerZoom > 0.01) {
        el.middleFingerOverlay.classList.add('active');

        // Calculate zoom center based on middle finger hand position
        if (state.middleFingerHand) {
            const middleTip = state.middleFingerHand[12];
            const wrist = state.middleFingerHand[0];

            // Center point between middle tip and wrist (mirrored)
            const centerX = (1 - (middleTip.x + wrist.x) / 2) * 100;
            const centerY = ((middleTip.y + wrist.y) / 2) * 100;

            // Apply zoom effect with easing
            const zoom = 1 + (state.middleFingerZoom * 0.8); // Max 1.8x zoom
            const opacity = state.middleFingerZoom;

            el.middleFingerOverlay.style.setProperty('--zoom', zoom);
            el.middleFingerOverlay.style.setProperty('--center-x', `${centerX}%`);
            el.middleFingerOverlay.style.setProperty('--center-y', `${centerY}%`);
            el.middleFingerOverlay.style.setProperty('--opacity', opacity);
        }

        // Show text
        if (el.middleFingerText) {
            el.middleFingerText.style.opacity = state.middleFingerZoom;
            el.middleFingerText.style.transform = `translate(-50%, -50%) scale(${0.8 + state.middleFingerZoom * 0.4})`;
        }
    } else {
        el.middleFingerOverlay.classList.remove('active');
        if (el.middleFingerText) {
            el.middleFingerText.style.opacity = 0;
        }
    }
}

// ============================================
// FACE INDICATOR
// ============================================
function updateFaceIndicator() {
    if (!el.faceIndicator) return;

    if (state.faceDetected) {
        el.faceIndicator.classList.add('detected');
    } else {
        el.faceIndicator.classList.remove('detected');
    }
}

// ============================================
// SECOND HAND POINTER
// ============================================
function updateSecondPointer(landmarks, timestamp) {
    if (!el.handPointer2) return;

    const indexTip = landmarks[8];
    const filters = state.handFilters[1];

    const rawX = (1 - indexTip.x) * 100;
    const rawY = indexTip.y * 100;

    const x = filters.filterX.filter(rawX, timestamp);
    const y = filters.filterY.filter(rawY, timestamp);

    state.cursors[1] = { x, y };

    el.handPointer2.classList.add('visible');
    el.handPointer2.style.left = `${x}%`;
    el.handPointer2.style.top = `${y}%`;

    // Check pinch for second hand
    const thumbTip = landmarks[4];
    const dx = thumbTip.x - indexTip.x;
    const dy = thumbTip.y - indexTip.y;
    const pinchDist = Math.sqrt(dx * dx + dy * dy);
    const isPinching = pinchDist < CONFIG.interaction.pinchThreshold;

    el.handPointer2.classList.toggle('pinching', isPinching);
}

// ============================================
// INTERACTION
// ============================================
function processInteraction(landmarks, timestamp) {
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const wrist = landmarks[0];
    const middleMcp = landmarks[9];

    // Hand size for adaptive thresholds
    state.handSize = Math.sqrt(
        Math.pow(middleMcp.x - wrist.x, 2) +
        Math.pow(middleMcp.y - wrist.y, 2)
    );

    // Cursor position (mirrored, filtered)
    const rawX = (1 - indexTip.x) * 100;
    const rawY = indexTip.y * 100;

    state.cursor.x = state.filterX.filter(rawX, timestamp);
    state.cursor.y = state.filterY.filter(rawY, timestamp);

    // Pinch detection with hysteresis
    const dx = thumbTip.x - indexTip.x;
    const dy = thumbTip.y - indexTip.y;
    const dz = (thumbTip.z || 0) - (indexTip.z || 0);
    state.pinchDistance = Math.sqrt(dx * dx + dy * dy + dz * dz);

    const threshold = state.isPinching
        ? CONFIG.interaction.pinchReleaseThreshold
        : CONFIG.interaction.pinchThreshold;

    const rawPinch = state.pinchDistance < threshold;
    const wasPinching = state.isPinching;
    state.isPinching = state.pinchStabilizer.update(rawPinch);

    // Update pointer
    updatePointer();

    // Find nearest object
    findNearestObject();

    // Grab/release
    if (state.isPinching && !wasPinching) {
        tryGrab();
    } else if (!state.isPinching && wasPinching) {
        releaseObject();
    }

    // Move grabbed
    if (state.grabbedObject) {
        moveGrabbed();
        checkDropZone();
    }
}

function updatePointer() {
    const ptr = el.handPointer;

    ptr.classList.add('visible');
    ptr.style.left = `${state.cursor.x}%`;
    ptr.style.top = `${state.cursor.y}%`;

    ptr.classList.toggle('pinching', state.isPinching);
    ptr.classList.toggle('near-object', state.nearestObject !== null && !state.grabbedObject);
}

function findNearestObject() {
    if (state.grabbedObject) return;

    let nearest = null;
    let minDist = CONFIG.interaction.grabRadius;

    const containerRect = el.cameraWindow.getBoundingClientRect();
    const cursorPx = {
        x: (state.cursor.x / 100) * containerRect.width,
        y: (state.cursor.y / 100) * containerRect.height
    };

    state.objects.forEach(obj => {
        if (obj.inZone) return;

        const objPx = {
            x: (obj.x / 100) * containerRect.width,
            y: (obj.y / 100) * containerRect.height
        };

        const dist = Math.sqrt(
            Math.pow(cursorPx.x - objPx.x, 2) +
            Math.pow(cursorPx.y - objPx.y, 2)
        );

        if (dist < minDist) {
            minDist = dist;
            nearest = obj;
        }
    });

    state.objects.forEach(obj => {
        obj.el.classList.toggle('hovering', obj === nearest);
    });

    state.nearestObject = nearest;
}

function tryGrab() {
    if (!state.nearestObject || state.nearestObject.inZone) return;

    const obj = state.nearestObject;
    obj.isGrabbed = true;
    state.grabbedObject = obj;

    state.grabOffset.x = obj.x - state.cursor.x;
    state.grabOffset.y = obj.y - state.cursor.y;

    obj.el.classList.add('grabbed');
    obj.el.classList.remove('hovering');

    showNotification('grabbed');
    updateObjectCard(obj.id, 'Held');
}

function moveGrabbed() {
    const obj = state.grabbedObject;
    if (!obj) return;

    let x = state.cursor.x + state.grabOffset.x;
    let y = state.cursor.y + state.grabOffset.y;

    x = Math.max(5, Math.min(95, x));
    y = Math.max(5, Math.min(95, y));

    obj.x = x;
    obj.y = y;
    obj.el.style.left = `${x}%`;
    obj.el.style.top = `${y}%`;
}

function checkDropZone() {
    const obj = state.grabbedObject;
    if (!obj) return;

    const dz = state.dropZone;
    const isOver =
        obj.x > dz.x - dz.w / 2 && obj.x < dz.x + dz.w / 2 &&
        obj.y > dz.y - dz.h / 2 && obj.y < dz.y + dz.h / 2;

    state.isOverDropZone = isOver;
    el.dropZone.classList.toggle('active', isOver);
}

function releaseObject() {
    const obj = state.grabbedObject;
    if (!obj) return;

    if (state.isOverDropZone && !obj.inZone) {
        dropInZone(obj);
    } else {
        obj.isGrabbed = false;
        obj.el.classList.remove('grabbed');
        updateObjectCard(obj.id, 'Free');
    }

    state.grabbedObject = null;
    state.isOverDropZone = false;
    el.dropZone.classList.remove('active');
}

function dropInZone(obj) {
    obj.inZone = true;
    obj.isGrabbed = false;
    obj.el.classList.remove('grabbed');
    obj.el.classList.add('in-zone');

    obj.x = state.dropZone.x;
    obj.y = state.dropZone.y;
    obj.el.style.left = `${obj.x}%`;
    obj.el.style.top = `${obj.y}%`;

    state.itemsInZone++;
    el.dropCount.textContent = `${state.itemsInZone} item${state.itemsInZone > 1 ? 's' : ''}`;
    el.dropZone.classList.add('has-items');

    showNotification('dropped');
    updateObjectCard(obj.id, 'Stored');
}

function resetObjects() {
    state.objects.forEach(obj => {
        obj.x = obj.origX;
        obj.y = obj.origY;
        obj.isGrabbed = false;
        obj.inZone = false;

        obj.el.style.left = `${obj.x}%`;
        obj.el.style.top = `${obj.y}%`;
        obj.el.classList.remove('grabbed', 'in-zone', 'hovering');

        updateObjectCard(obj.id, 'Free');
    });

    state.grabbedObject = null;
    state.itemsInZone = 0;
    el.dropCount.textContent = '0 items';
    el.dropZone.classList.remove('has-items', 'active');

    resetFilters();
}

// ============================================
// UI
// ============================================
function updateUI(handCount, faceCount = 0) {
    // Pills
    el.fpsPill.querySelector('.pill-value').textContent = state.fps;
    el.handsPill.querySelector('.pill-value').textContent = handCount;

    let gestureText = state.middleFingerDetected ? '🖕 Detected!' :
        state.grabbedObject ? 'Holding' :
            state.isPinching ? 'Pinching' :
                state.nearestObject ? 'Hover' : 'Ready';

    el.gesturePill.querySelector('.pill-value').textContent = gestureText;

    // Add visual feedback for middle finger
    if (state.middleFingerDetected) {
        el.gesturePill.classList.add('warning');
    } else {
        el.gesturePill.classList.remove('warning');
    }

    // Data
    el.cursorData.textContent = `${Math.round(state.cursor.x)}, ${Math.round(state.cursor.y)}`;

    const pinchPct = Math.round((1 - state.pinchDistance / 0.15) * 100);
    el.pinchData.textContent = `${Math.max(0, Math.min(100, pinchPct))}%`;
    el.pinchData.style.color = state.isPinching ? 'var(--accent-cyan)' : 'var(--text-tertiary)';

    el.holdingData.textContent = state.grabbedObject?.type || 'None';
}

function updateObjectCard(objId, statusText) {
    const card = el.objectsGrid.querySelector(`[data-obj="${objId}"]`);
    if (!card) return;

    const statusEl = card.querySelector('.obj-status');
    if (statusEl) statusEl.textContent = statusText;

    card.classList.remove('active', 'in-zone');
    if (statusText === 'Held') card.classList.add('active');
    if (statusText === 'Stored') card.classList.add('in-zone');
}

function showNotification(type) {
    const notif = type === 'grabbed' ? el.notifGrabbed : el.notifDropped;

    notif.classList.add('show');
    setTimeout(() => notif.classList.remove('show'), 1500);
}

// ============================================
// START
// ============================================
init();
