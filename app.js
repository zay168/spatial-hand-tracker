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
// SKELETON RENDERING
// ============================================
function renderSkeleton(landmarks, handIndex = 0) {
    const ctx = state.ctx;
    const w = state.canvasW;
    const h = state.canvasH;
    
    // Different color schemes for different hands
    const colorSchemes = [
        ['#ff9500', '#5ac8fa', '#bf5af2', '#ff375f', '#30d158', '#ffffff'],
        ['#00ff88', '#ff6b9d', '#ffbe0b', '#8338ec', '#3a86ff', '#e0e0e0']
    ];
    const colors = colorSchemes[handIndex % 2];

    // Bones by color
    for (let c = 0; c < colors.length; c++) {
        ctx.beginPath();
        ctx.strokeStyle = colors[c];
        ctx.lineWidth = CONFIG.rendering.skeleton.boneWidth;
        ctx.lineCap = 'round';

        for (const [start, end, col] of HAND_CONNECTIONS) {
            if (col !== c) continue;
            ctx.moveTo(landmarks[start].x * w, landmarks[start].y * h);
            ctx.lineTo(landmarks[end].x * w, landmarks[end].y * h);
        }
        ctx.stroke();
    }

    // Joints
    for (let i = 0; i < 21; i++) {
        const x = landmarks[i].x * w;
        const y = landmarks[i].y * h;
        const colIdx = i <= 4 ? 0 : i <= 8 ? 1 : i <= 12 ? 2 : i <= 16 ? 3 : 4;

        ctx.fillStyle = colors[colIdx];
        ctx.beginPath();
        ctx.arc(x, y, CONFIG.rendering.skeleton.jointRadius, 0, Math.PI * 2);
        ctx.fill();
    }
}

// ============================================
// FACE RENDERING
// ============================================
function renderFace(landmarks) {
    const ctx = state.ctx;
    const w = state.canvasW;
    const h = state.canvasH;
    
    ctx.save();
    ctx.translate(state.canvasW, 0);
    ctx.scale(-1, 1);
    
    // Draw face mesh outline (simplified - just key points)
    const keyPoints = [10, 152, 234, 454, 21, 251, 33, 263, 61, 291, 199]; // Key face landmarks
    
    ctx.strokeStyle = 'rgba(90, 200, 250, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    
    // Draw oval around face
    if (landmarks.length > 300) {
        // Get face boundary points for oval
        const faceOvalPoints = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
        
        ctx.beginPath();
        for (let i = 0; i < faceOvalPoints.length; i++) {
            const point = landmarks[faceOvalPoints[i]];
            const x = point.x * w;
            const y = point.y * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
    }
    
    ctx.setLineDash([]);
    
    // Draw eyes
    const leftEye = [33, 160, 158, 133, 153, 144];
    const rightEye = [362, 385, 387, 263, 373, 380];
    
    ctx.strokeStyle = 'rgba(90, 200, 250, 0.6)';
    ctx.lineWidth = 1.5;
    
    [leftEye, rightEye].forEach(eye => {
        ctx.beginPath();
        eye.forEach((idx, i) => {
            if (idx < landmarks.length) {
                const point = landmarks[idx];
                if (i === 0) ctx.moveTo(point.x * w, point.y * h);
                else ctx.lineTo(point.x * w, point.y * h);
            }
        });
        ctx.closePath();
        ctx.stroke();
    });
    
    ctx.restore();
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
