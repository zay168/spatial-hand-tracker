/**
 * Spatial Hand Interaction - Vision Pro Style
 * PRECISION ENHANCED with Kalman Filter & Advanced Algorithms
 */

import { HandLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest';

// ============================================
// PRECISION CONFIGURATION
// ============================================
const CONFIG = {
    // MediaPipe settings - Higher confidence for precision
    numHands: 1,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.7,
    minTrackingConfidence: 0.7,

    // Interaction thresholds (normalized to hand size)
    pinchThresholdRatio: 0.25,      // Relative to hand size
    pinchReleaseRatio: 0.35,        // Hysteresis: release threshold > grab
    grabRadiusRatio: 2.5,           // Object grab radius relative to hand size

    // Precision settings
    kalman: {
        processNoise: 0.01,         // Lower = smoother but more lag
        measurementNoise: 0.1,      // Higher = more trust in filtering
        initialUncertainty: 1.0
    },

    // Gesture stability
    gestureFramesRequired: 3,       // Frames to confirm gesture change
    gestureFramesRelease: 2,        // Frames to confirm release

    // Cursor
    useMidpointCursor: true,        // Use thumb-index midpoint for pinch
    predictiveFrames: 2,            // Frames to predict ahead

    // Skeleton
    skeleton: {
        jointRadius: 4,
        boneWidth: 2,
        colors: ['#ff6b00', '#00f5ff', '#bf00ff', '#ff00f5', '#39ff14', '#ffffff']
    },

    // Performance
    uiUpdateInterval: 80
};

// Hand connections
const HAND_CONNECTIONS = [
    [0, 1, 0], [1, 2, 0], [2, 3, 0], [3, 4, 0],
    [0, 5, 1], [5, 6, 1], [6, 7, 1], [7, 8, 1],
    [0, 9, 2], [9, 10, 2], [10, 11, 2], [11, 12, 2],
    [0, 13, 3], [13, 14, 3], [14, 15, 3], [15, 16, 3],
    [0, 17, 4], [17, 18, 4], [18, 19, 4], [19, 20, 4],
    [5, 9, 5], [9, 13, 5], [13, 17, 5]
];

// ============================================
// KALMAN FILTER CLASS
// ============================================
class KalmanFilter {
    constructor(processNoise = 0.01, measurementNoise = 0.1, initialUncertainty = 1.0) {
        this.Q = processNoise;      // Process noise covariance
        this.R = measurementNoise;  // Measurement noise covariance
        this.P = initialUncertainty; // Estimation error covariance
        this.X = 0;                  // State estimate
        this.K = 0;                  // Kalman gain
        this.initialized = false;
    }

    filter(measurement) {
        if (!this.initialized) {
            this.X = measurement;
            this.initialized = true;
            return measurement;
        }

        // Prediction
        this.P = this.P + this.Q;

        // Update
        this.K = this.P / (this.P + this.R);
        this.X = this.X + this.K * (measurement - this.X);
        this.P = (1 - this.K) * this.P;

        return this.X;
    }

    reset() {
        this.initialized = false;
        this.P = CONFIG.kalman.initialUncertainty;
    }
}

// ============================================
// VELOCITY PREDICTOR CLASS
// ============================================
class VelocityPredictor {
    constructor(historySize = 5) {
        this.history = [];
        this.historySize = historySize;
        this.lastTimestamp = 0;
    }

    update(x, y, timestamp) {
        const dt = this.lastTimestamp > 0 ? timestamp - this.lastTimestamp : 16;
        this.lastTimestamp = timestamp;

        if (this.history.length > 0) {
            const last = this.history[this.history.length - 1];
            const vx = (x - last.x) / dt;
            const vy = (y - last.y) / dt;

            this.history.push({ x, y, vx, vy, dt });
        } else {
            this.history.push({ x, y, vx: 0, vy: 0, dt });
        }

        if (this.history.length > this.historySize) {
            this.history.shift();
        }
    }

    predict(frames = 1) {
        if (this.history.length < 2) {
            const last = this.history[this.history.length - 1] || { x: 0, y: 0 };
            return { x: last.x, y: last.y };
        }

        // Weighted average velocity
        let avgVx = 0, avgVy = 0, totalWeight = 0;

        for (let i = 0; i < this.history.length; i++) {
            const weight = (i + 1) / this.history.length; // More recent = higher weight
            avgVx += this.history[i].vx * weight;
            avgVy += this.history[i].vy * weight;
            totalWeight += weight;
        }

        avgVx /= totalWeight;
        avgVy /= totalWeight;

        const last = this.history[this.history.length - 1];
        const predictedDt = 16 * frames; // Assume 60fps

        return {
            x: last.x + avgVx * predictedDt,
            y: last.y + avgVy * predictedDt
        };
    }

    reset() {
        this.history = [];
        this.lastTimestamp = 0;
    }
}

// ============================================
// GESTURE STABILIZER CLASS
// ============================================
class GestureStabilizer {
    constructor(requiredFrames = 3, releaseFrames = 2) {
        this.requiredFrames = requiredFrames;
        this.releaseFrames = releaseFrames;
        this.currentState = false;
        this.pendingState = false;
        this.frameCount = 0;
    }

    update(rawState) {
        if (rawState !== this.pendingState) {
            // State change detected
            this.pendingState = rawState;
            this.frameCount = 1;
        } else {
            // Same state
            this.frameCount++;
        }

        // Check if we should transition
        const threshold = this.currentState ? this.releaseFrames : this.requiredFrames;

        if (this.frameCount >= threshold && this.pendingState !== this.currentState) {
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
// ONE EURO FILTER (Alternative smooth filter)
// ============================================
class OneEuroFilter {
    constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        this.xPrev = null;
        this.dxPrev = 0;
        this.tPrev = null;
    }

    alpha(cutoff, dt) {
        const tau = 1.0 / (2 * Math.PI * cutoff);
        return 1.0 / (1.0 + tau / dt);
    }

    filter(x, t) {
        if (this.xPrev === null) {
            this.xPrev = x;
            this.tPrev = t;
            return x;
        }

        const dt = Math.max((t - this.tPrev) / 1000, 0.001);
        this.tPrev = t;

        // Derivative
        const dx = (x - this.xPrev) / dt;
        const edx = this.alpha(this.dCutoff, dt) * dx + (1 - this.alpha(this.dCutoff, dt)) * this.dxPrev;
        this.dxPrev = edx;

        // Adaptive cutoff
        const cutoff = this.minCutoff + this.beta * Math.abs(edx);

        // Filter
        const result = this.alpha(cutoff, dt) * x + (1 - this.alpha(cutoff, dt)) * this.xPrev;
        this.xPrev = result;

        return result;
    }

    reset() {
        this.xPrev = null;
        this.dxPrev = 0;
        this.tPrev = null;
    }
}

// ============================================
// STATE
// ============================================
const state = {
    handLandmarker: null,
    isRunning: false,

    // FPS tracking
    frameCount: 0,
    fps: 0,
    fpsFrameCount: 0,
    lastFpsUpdate: 0,
    lastUIUpdate: 0,

    // Canvas
    skeletonCtx: null,
    canvasWidth: 0,
    canvasHeight: 0,

    // Precision filters
    kalmanX: new KalmanFilter(CONFIG.kalman.processNoise, CONFIG.kalman.measurementNoise),
    kalmanY: new KalmanFilter(CONFIG.kalman.processNoise, CONFIG.kalman.measurementNoise),
    oneEuroX: new OneEuroFilter(1.0, 0.007, 1.0),
    oneEuroY: new OneEuroFilter(1.0, 0.007, 1.0),
    velocityPredictor: new VelocityPredictor(5),
    pinchStabilizer: new GestureStabilizer(CONFIG.gestureFramesRequired, CONFIG.gestureFramesRelease),

    // Hand tracking
    cursor: { x: 0, y: 0, rawX: 0, rawY: 0 },
    isPinching: false,
    pinchDistance: 0,
    handSize: 0,           // Dynamic hand size for adaptive thresholds
    handConfidence: 0,

    // Objects
    objects: [],
    grabbedObject: null,
    grabOffset: { x: 0, y: 0 },
    nearestObject: null,
    nearestDistance: Infinity,

    // Drop zone
    dropZone: { x: 0, y: 0, width: 0, height: 0 },
    isOverDropZone: false,
    objectsInBox: 0
};

// ============================================
// DOM ELEMENTS
// ============================================
const $ = id => document.getElementById(id);
const elements = {
    webcam: $('webcam'),
    skeletonCanvas: $('skeletonCanvas'),
    objectsLayer: $('objectsLayer'),
    videoContainer: $('videoContainer'),
    loadingOverlay: $('loadingOverlay'),
    loadingStatus: $('loadingStatus'),
    permissionOverlay: $('permissionOverlay'),
    startButton: $('startButton'),
    resetButton: $('resetButton'),
    handCursor: $('handCursor'),
    dropZone: $('dropZone'),
    gestureFeedback: $('gestureFeedback'),
    grabFeedback: $('grabFeedback'),
    dropSuccess: $('dropSuccess'),
    fpsValue: $('fpsCounter')?.querySelector('.status-value'),
    handsValue: $('handsDetected')?.querySelector('.status-value'),
    gestureValue: $('gestureStatus')?.querySelector('.status-value'),
    objectsInBoxValue: $('objectsInBox')?.querySelector('.status-value'),
    cursorX: $('cursorX'),
    cursorY: $('cursorY'),
    pinchValue: $('pinchValue'),
    grabbedObj: $('grabbedObj'),
    objectsList: $('objectsList')
};

// ============================================
// MATH UTILITIES
// ============================================
function distance3D(p1, p2) {
    return Math.sqrt(
        Math.pow(p1.x - p2.x, 2) +
        Math.pow(p1.y - p2.y, 2) +
        Math.pow((p1.z || 0) - (p2.z || 0), 2)
    );
}

function midpoint(p1, p2) {
    return {
        x: (p1.x + p2.x) / 2,
        y: (p1.y + p2.y) / 2,
        z: ((p1.z || 0) + (p2.z || 0)) / 2
    };
}

function calculateHandSize(landmarks) {
    // Use wrist to middle finger MCP as base reference
    const wrist = landmarks[0];
    const middleMCP = landmarks[9];
    const middleTip = landmarks[12];

    // Palm size (wrist to middle MCP)
    const palmSize = distance3D(wrist, middleMCP);

    // Full hand size (wrist to middle tip)
    const fullSize = distance3D(wrist, middleTip);

    // Use average for stability
    return (palmSize + fullSize) / 2;
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

// ============================================
// INITIALIZATION
// ============================================
async function init() {
    try {
        elements.loadingStatus.textContent = 'Loading AI model...';

        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );

        elements.loadingStatus.textContent = 'Initializing precision tracking...';

        state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numHands: CONFIG.numHands,
            minHandDetectionConfidence: CONFIG.minHandDetectionConfidence,
            minHandPresenceConfidence: CONFIG.minHandPresenceConfidence,
            minTrackingConfidence: CONFIG.minTrackingConfidence
        });

        setupCanvas();
        initializeObjects();
        setupDropZone();
        setupEventListeners();

        elements.loadingStatus.textContent = 'Ready!';
        elements.loadingOverlay.classList.add('hidden');

    } catch (e) {
        console.error('Init error:', e);
        elements.loadingStatus.textContent = `Error: ${e.message}`;
    }
}

function setupCanvas() {
    state.skeletonCtx = elements.skeletonCanvas.getContext('2d', {
        alpha: true,
        desynchronized: true
    });
}

function initializeObjects() {
    const objectElements = elements.objectsLayer.querySelectorAll('.spatial-object');

    objectElements.forEach(el => {
        state.objects.push({
            id: el.id,
            element: el,
            x: parseFloat(el.style.left) || 15,
            y: parseFloat(el.style.top) || 30,
            originalX: parseFloat(el.style.left) || 15,
            originalY: parseFloat(el.style.top) || 30,
            type: el.dataset.type,
            isGrabbed: false,
            isInBox: false,
            // Smooth movement for objects too
            smoothX: parseFloat(el.style.left) || 15,
            smoothY: parseFloat(el.style.top) || 30
        });
    });
}

function setupDropZone() {
    updateDropZonePosition();
    window.addEventListener('resize', updateDropZonePosition);
}

function updateDropZonePosition() {
    const rect = elements.dropZone.getBoundingClientRect();
    const containerRect = elements.videoContainer.getBoundingClientRect();

    state.dropZone = {
        x: ((rect.left - containerRect.left + rect.width / 2) / containerRect.width) * 100,
        y: ((rect.top - containerRect.top + rect.height / 2) / containerRect.height) * 100,
        width: (rect.width / containerRect.width) * 100,
        height: (rect.height / containerRect.height) * 100
    };
}

function setupEventListeners() {
    elements.startButton.onclick = startCamera;
    elements.resetButton.onclick = resetAllObjects;
    window.onresize = () => {
        resizeCanvas();
        updateDropZonePosition();
    };
}

// ============================================
// CAMERA
// ============================================
async function startCamera() {
    try {
        elements.permissionOverlay.classList.add('hidden');
        elements.loadingOverlay.classList.remove('hidden');
        elements.loadingStatus.textContent = 'Accessing camera...';

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user',
                frameRate: { ideal: 30 }
            }
        });

        elements.webcam.srcObject = stream;
        await elements.webcam.play();

        resizeCanvas();
        elements.loadingOverlay.classList.add('hidden');

        state.isRunning = true;
        requestAnimationFrame(processFrame);

    } catch (e) {
        console.error('Camera error:', e);
        elements.loadingStatus.textContent = `Camera error: ${e.message}`;
        elements.permissionOverlay.classList.remove('hidden');
    }
}

function resizeCanvas() {
    const container = elements.webcam.parentElement;
    const w = container.clientWidth;
    const h = container.clientHeight;

    elements.skeletonCanvas.width = w;
    elements.skeletonCanvas.height = h;

    state.canvasWidth = w;
    state.canvasHeight = h;
}

// ============================================
// FRAME PROCESSING
// ============================================
function processFrame(timestamp) {
    if (!state.isRunning) return;

    state.frameCount++;

    // FPS calculation
    state.fpsFrameCount++;
    if (timestamp - state.lastFpsUpdate >= 1000) {
        state.fps = state.fpsFrameCount;
        state.fpsFrameCount = 0;
        state.lastFpsUpdate = timestamp;
    }

    // Hand detection
    const results = state.handLandmarker.detectForVideo(elements.webcam, timestamp);

    // Clear canvas
    const ctx = state.skeletonCtx;
    ctx.clearRect(0, 0, state.canvasWidth, state.canvasHeight);

    const numHands = results.landmarks?.length || 0;

    if (numHands > 0) {
        const landmarks = results.landmarks[0];

        // Store hand confidence
        state.handConfidence = results.handednesses?.[0]?.[0]?.score || 0;

        // Mirror transform for skeleton
        ctx.save();
        ctx.translate(state.canvasWidth, 0);
        ctx.scale(-1, 1);

        renderSkeleton(ctx, landmarks);

        ctx.restore();

        // Process hand interaction with precision algorithms
        processHandInteractionPrecise(landmarks, timestamp);

    } else {
        // No hand detected - reset filters
        elements.handCursor.classList.remove('visible', 'pinching', 'near-object');

        if (state.grabbedObject) {
            releaseObject();
        }

        // Reset filters for next detection
        resetFilters();
    }

    // Throttled UI update
    if (timestamp - state.lastUIUpdate >= CONFIG.uiUpdateInterval) {
        updateUI(numHands);
        state.lastUIUpdate = timestamp;
    }

    requestAnimationFrame(processFrame);
}

function resetFilters() {
    state.kalmanX.reset();
    state.kalmanY.reset();
    state.oneEuroX.reset();
    state.oneEuroY.reset();
    state.velocityPredictor.reset();
    state.pinchStabilizer.reset();
}

// ============================================
// SKELETON RENDERING
// ============================================
function renderSkeleton(ctx, landmarks) {
    const w = state.canvasWidth;
    const h = state.canvasHeight;

    // Draw bones by color
    for (let c = 0; c < 6; c++) {
        ctx.beginPath();
        ctx.strokeStyle = CONFIG.skeleton.colors[c];
        ctx.lineWidth = CONFIG.skeleton.boneWidth;
        ctx.lineCap = 'round';

        for (const [start, end, color] of HAND_CONNECTIONS) {
            if (color !== c) continue;

            const x1 = landmarks[start].x * w;
            const y1 = landmarks[start].y * h;
            const x2 = landmarks[end].x * w;
            const y2 = landmarks[end].y * h;

            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
        }
        ctx.stroke();
    }

    // Draw joints
    for (let i = 0; i < 21; i++) {
        const x = landmarks[i].x * w;
        const y = landmarks[i].y * h;
        const colorIndex = i <= 4 ? 0 : i <= 8 ? 1 : i <= 12 ? 2 : i <= 16 ? 3 : 4;

        ctx.fillStyle = CONFIG.skeleton.colors[colorIndex];
        ctx.beginPath();
        ctx.arc(x, y, CONFIG.skeleton.jointRadius, 0, Math.PI * 2);
        ctx.fill();
    }

    // Draw pinch point indicator
    if (state.isPinching) {
        const thumbTip = landmarks[4];
        const indexTip = landmarks[8];
        const pinchPoint = midpoint(thumbTip, indexTip);

        ctx.fillStyle = '#00f5ff';
        ctx.beginPath();
        ctx.arc(pinchPoint.x * w, pinchPoint.y * h, 8, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

// ============================================
// PRECISION HAND INTERACTION
// ============================================
function processHandInteractionPrecise(landmarks, timestamp) {
    const indexTip = landmarks[8];
    const thumbTip = landmarks[4];

    // Calculate dynamic hand size for adaptive thresholds
    state.handSize = calculateHandSize(landmarks);

    // Calculate raw cursor position
    let rawX, rawY;

    if (CONFIG.useMidpointCursor && state.isPinching) {
        // Use midpoint between thumb and index when pinching
        const mid = midpoint(thumbTip, indexTip);
        rawX = (1 - mid.x) * 100;
        rawY = mid.y * 100;
    } else {
        // Use index finger tip
        rawX = (1 - indexTip.x) * 100;
        rawY = indexTip.y * 100;
    }

    state.cursor.rawX = rawX;
    state.cursor.rawY = rawY;

    // Apply One Euro Filter (adaptive smoothing - less lag during fast movement)
    const filteredX = state.oneEuroX.filter(rawX, timestamp);
    const filteredY = state.oneEuroY.filter(rawY, timestamp);

    // Update velocity predictor
    state.velocityPredictor.update(filteredX, filteredY, timestamp);

    // Apply predictive offset for reduced perceived latency
    if (CONFIG.predictiveFrames > 0 && state.grabbedObject) {
        const predicted = state.velocityPredictor.predict(CONFIG.predictiveFrames);
        state.cursor.x = lerp(filteredX, predicted.x, 0.3);
        state.cursor.y = lerp(filteredY, predicted.y, 0.3);
    } else {
        state.cursor.x = filteredX;
        state.cursor.y = filteredY;
    }

    // Calculate pinch distance with 3D consideration
    state.pinchDistance = distance3D(thumbTip, indexTip);

    // Adaptive thresholds based on hand size
    const pinchThreshold = state.handSize * CONFIG.pinchThresholdRatio;
    const releaseThreshold = state.handSize * CONFIG.pinchReleaseRatio;

    // Raw pinch state with hysteresis
    const rawPinchState = state.isPinching
        ? state.pinchDistance < releaseThreshold  // Already pinching: use higher threshold to release
        : state.pinchDistance < pinchThreshold;   // Not pinching: use lower threshold to start

    // Apply gesture stabilizer for flicker-free detection
    const wasPinching = state.isPinching;
    state.isPinching = state.pinchStabilizer.update(rawPinchState);

    // Update hand cursor
    updateHandCursor();

    // Find nearest object with precision
    findNearestObjectPrecise();

    // Handle grab/release with confirmation
    if (state.isPinching && !wasPinching) {
        tryGrabObject();
    } else if (!state.isPinching && wasPinching) {
        releaseObject();
    }

    // Move grabbed object with smooth interpolation
    if (state.grabbedObject) {
        moveGrabbedObjectSmooth();
        checkDropZone();
    }
}

function updateHandCursor() {
    const cursor = elements.handCursor;

    cursor.classList.add('visible');
    cursor.style.left = `${state.cursor.x}%`;
    cursor.style.top = `${state.cursor.y}%`;

    cursor.classList.toggle('pinching', state.isPinching);
    cursor.classList.toggle('near-object', state.nearestObject !== null && !state.grabbedObject);
}

function findNearestObjectPrecise() {
    if (state.grabbedObject) return;

    // Adaptive grab radius based on hand size
    const grabRadius = state.handSize * CONFIG.grabRadiusRatio * state.canvasWidth;

    let nearest = null;
    let minDist = grabRadius;

    const containerRect = elements.videoContainer.getBoundingClientRect();
    const cursorPx = {
        x: (state.cursor.x / 100) * containerRect.width,
        y: (state.cursor.y / 100) * containerRect.height
    };

    state.objects.forEach(obj => {
        if (obj.isInBox) return;

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

    // Update hover states
    state.objects.forEach(obj => {
        obj.element.classList.toggle('hovering', obj === nearest);
    });

    state.nearestObject = nearest;
    state.nearestDistance = minDist;
}

function tryGrabObject() {
    if (!state.nearestObject || state.nearestObject.isInBox) return;

    const obj = state.nearestObject;
    obj.isGrabbed = true;
    state.grabbedObject = obj;

    // Calculate grab offset
    state.grabOffset.x = obj.x - state.cursor.x;
    state.grabOffset.y = obj.y - state.cursor.y;

    // Initialize smooth position
    obj.smoothX = obj.x;
    obj.smoothY = obj.y;

    // Visual feedback
    obj.element.classList.add('grabbed');
    obj.element.classList.remove('hovering');

    showFeedback('grab');
    updateObjectStatus(obj.id, 'Grabbed');
}

function moveGrabbedObjectSmooth() {
    if (!state.grabbedObject) return;

    const obj = state.grabbedObject;

    // Target position
    let targetX = state.cursor.x + state.grabOffset.x;
    let targetY = state.cursor.y + state.grabOffset.y;

    // Clamp to container
    targetX = Math.max(5, Math.min(95, targetX));
    targetY = Math.max(5, Math.min(95, targetY));

    // Smooth interpolation for object movement
    obj.smoothX = lerp(obj.smoothX, targetX, 0.4);
    obj.smoothY = lerp(obj.smoothY, targetY, 0.4);

    obj.x = obj.smoothX;
    obj.y = obj.smoothY;

    // Update DOM
    obj.element.style.left = `${obj.x}%`;
    obj.element.style.top = `${obj.y}%`;
}

function checkDropZone() {
    if (!state.grabbedObject) return;

    const obj = state.grabbedObject;
    const dz = state.dropZone;

    // Check if object center is over drop zone
    const isOver = obj.x > dz.x - dz.width / 2 &&
        obj.x < dz.x + dz.width / 2 &&
        obj.y > dz.y - dz.height / 2 &&
        obj.y < dz.y + dz.height / 2;

    state.isOverDropZone = isOver;
    elements.dropZone.classList.toggle('active', isOver);
}

function releaseObject() {
    if (!state.grabbedObject) return;

    const obj = state.grabbedObject;

    if (state.isOverDropZone && !obj.isInBox) {
        dropInBox(obj);
    } else {
        obj.isGrabbed = false;
        obj.element.classList.remove('grabbed');
        updateObjectStatus(obj.id, 'Free');
    }

    state.grabbedObject = null;
    state.isOverDropZone = false;
    elements.dropZone.classList.remove('active');
    hideFeedback('grab');
}

function dropInBox(obj) {
    obj.isInBox = true;
    obj.isGrabbed = false;
    obj.element.classList.remove('grabbed');
    obj.element.classList.add('in-box');

    // Animate to drop zone center
    obj.x = state.dropZone.x;
    obj.y = state.dropZone.y;
    obj.smoothX = obj.x;
    obj.smoothY = obj.y;
    obj.element.style.left = `${obj.x}%`;
    obj.element.style.top = `${obj.y}%`;

    state.objectsInBox++;
    elements.dropZone.classList.add('has-objects');

    showFeedback('drop');
    updateObjectStatus(obj.id, 'In Box');
}

function resetAllObjects() {
    state.objects.forEach(obj => {
        obj.x = obj.originalX;
        obj.y = obj.originalY;
        obj.smoothX = obj.originalX;
        obj.smoothY = obj.originalY;
        obj.isGrabbed = false;
        obj.isInBox = false;

        obj.element.style.left = `${obj.x}%`;
        obj.element.style.top = `${obj.y}%`;
        obj.element.classList.remove('grabbed', 'in-box', 'hovering');

        updateObjectStatus(obj.id, 'Free');
    });

    state.grabbedObject = null;
    state.objectsInBox = 0;
    elements.dropZone.classList.remove('has-objects', 'active');

    // Reset filters
    resetFilters();
}

// ============================================
// UI UPDATES
// ============================================
function updateUI(handCount) {
    if (elements.fpsValue) elements.fpsValue.textContent = state.fps;
    if (elements.handsValue) elements.handsValue.textContent = handCount;
    if (elements.objectsInBoxValue) elements.objectsInBoxValue.textContent = state.objectsInBox;

    if (elements.gestureValue) {
        if (state.grabbedObject) {
            elements.gestureValue.textContent = 'HOLDING';
        } else if (state.isPinching) {
            elements.gestureValue.textContent = 'PINCH';
        } else if (state.nearestObject) {
            elements.gestureValue.textContent = 'HOVER';
        } else {
            elements.gestureValue.textContent = '--';
        }
    }

    if (elements.cursorX) elements.cursorX.textContent = Math.round(state.cursor.x);
    if (elements.cursorY) elements.cursorY.textContent = Math.round(state.cursor.y);
    if (elements.pinchValue) {
        const pinchPct = Math.round((1 - state.pinchDistance / (state.handSize * 0.5)) * 100);
        elements.pinchValue.textContent = `${Math.max(0, pinchPct)}%`;
        elements.pinchValue.style.color = state.isPinching ? '#00f5ff' : '#ff453a';
    }
    if (elements.grabbedObj) {
        elements.grabbedObj.textContent = state.grabbedObject?.type || '--';
    }
}

function updateObjectStatus(objId, status) {
    const statusEl = elements.objectsList?.querySelector(`[data-for="${objId}"]`);
    if (!statusEl) return;

    const stateEl = statusEl.querySelector('.obj-state');
    if (stateEl) stateEl.textContent = status;

    statusEl.classList.remove('grabbed', 'in-box');
    if (status === 'Grabbed') statusEl.classList.add('grabbed');
    if (status === 'In Box') statusEl.classList.add('in-box');
}

function showFeedback(type) {
    if (type === 'grab') {
        elements.grabFeedback.classList.add('active');
    } else if (type === 'drop') {
        elements.dropSuccess.classList.add('active');
        setTimeout(() => {
            elements.dropSuccess.classList.remove('active');
        }, 1500);
    }
}

function hideFeedback(type) {
    if (type === 'grab') {
        elements.grabFeedback.classList.remove('active');
    }
}

// ============================================
// START
// ============================================
init();
