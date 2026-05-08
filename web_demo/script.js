// 1. Configuration matches Python keys
const CONFIDENCE_THRESHOLD = 0.7;
const CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const statusDiv = document.getElementById('status');
const predictionText = document.getElementById('prediction-text');
const confidenceText = document.getElementById('confidence-text');

let model;

// 2. Load Model
async function loadModel() {
    try {
        statusDiv.innerText = 'Loading model...';
        model = await tf.loadLayersModel('./model/model.json');
        statusDiv.className = 'status-success';
        statusDiv.innerText = 'Model loaded! Starting camera...';
        startCamera();
    } catch (error) {
        console.error(error);
        statusDiv.className = 'status-error';
        statusDiv.innerText = 'Error loading model: ' + error.message;
    }
}

// 3. Preprocessing Logic (Matches Python `extract_landmarks`)
function extractLandmarks(handLandmarks, handLabel) {
    const landmarks = [];

    // 1. Coordinates: x, y, z
    for (const landmark of handLandmarks) {
        // MIRROR LOGIC:
        // Python script uses cv2.flip(image, 1).
        // This inverts the X coordinate.
        // We must do the same here to match the model training data.
        const mirroredX = 1.0 - landmark.x;

        landmarks.push(mirroredX, landmark.y, landmark.z);
    }

    // 2. Hand Code: Right=1, Left=-1
    // Matches Python inisiasi logic: 1 if hand_type == "Right" else -1
    const handCode = (handLabel === 'Right') ? 1 : -1;

    // Insert handCode at the beginning
    landmarks.unshift(handCode);

    return landmarks;
}

// 4. MediaPipe Hands
function onResults(results) {
    // Save UI
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Draw Camera Feed
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    // Logic Containers (Zeros [64])
    let leftHandData = new Array(64).fill(0);
    let rightHandData = new Array(64).fill(0);

    if (results.multiHandLandmarks && results.multiHandedness) {
        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            const landmarks = results.multiHandLandmarks[i];
            const classification = results.multiHandedness[i];

            // Note on Classification:
            // Python training data used cv2.flip(1), which makes a Right hand look like a Left hand.
            // So Physical Right Hand -> Python detected "Left".
            // JS MediaPipe detects Physical Right Hand as "Right".
            // To match the model input, we must SWAP the labels.

            const originalLabel = classification.label;
            const effectiveLabel = (originalLabel === 'Right') ? 'Left' : 'Right';

            // Extract Features using the EFFECTIVE label
            const feats = extractLandmarks(landmarks, effectiveLabel);

            // Slot Data using the EFFECTIVE label
            if (effectiveLabel === 'Left') {
                leftHandData = feats;
            } else {
                rightHandData = feats;
            }

            // Draw Landmarks (Visuals can stay true to original or swap, let's keep visual generic)
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
            drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });
        }

        // Combine Data: Left + Right (Length 128)
        const combinedData = leftHandData.concat(rightHandData);

        // Slice to 126 (Match Training Input)
        const inputFeatures = combinedData.slice(0, 126);

        // Predict
        predict(inputFeatures);

    } else {
        predictionText.innerText = "Waiting for hands...";
        predictionText.className = "prediction-text";
        confidenceText.innerText = "Confidence: 0%";
    }

    canvasCtx.restore();
}

async function predict(inputFeatures) {
    if (!model) return;

    // Tensor [1, 126]
    const inputTensor = tf.tensor2d([inputFeatures], [1, 126]);

    const prediction = model.predict(inputTensor);
    const result = await prediction.data(); // Float32Array

    // Dispose tensor
    inputTensor.dispose();

    // Find Max
    let maxScore = -1;
    let maxIndex = -1;

    for (let i = 0; i < result.length; i++) {
        if (result[i] > maxScore) {
            maxScore = result[i];
            maxIndex = i;
        }
    }

    const label = CLASSES[maxIndex];
    const confidencePct = (maxScore * 100).toFixed(1);

    // Update UI
    if (maxScore > CONFIDENCE_THRESHOLD) {
        predictionText.innerText = `${label} (${confidencePct}%)`;
        predictionText.className = "prediction-text high-confidence";
    } else {
        predictionText.innerText = `Low Conf: ${label}`;
        predictionText.className = "prediction-text low-confidence";
    }
    confidenceText.innerText = `Confidence: ${confidencePct}%`;
}

// 5. Initialize MediaPipe
const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

// 6. Camera Utils
function startCamera() {
    const camera = new Camera(videoElement, {
        onFrame: async () => {
            await hands.send({ image: videoElement });
        },
        width: 640,
        height: 480
    });
    camera.start();
}

// Trigger Init
loadModel();
