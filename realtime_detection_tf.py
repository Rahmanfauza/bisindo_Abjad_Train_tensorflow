import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# 1. Configuration
MODEL_PATH = 'bisindo_mlp_model.keras'
CLASSES_PATH = 'classes.npy'
CONFIDENCE_THRESHOLD = 0.7

# 2. Load Model and Classes
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH)
    print(f"Model loaded. Classes: {classes}")
except Exception as e:
    print(f"Error loading model or classes: {e}")
    exit()

# 3. Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

def extract_landmarks(hand_landmarks, hand_type):
    """
    Refactored to match `inisiasi cllass.py` EXACTLY.
    Returns: [hand_code, x1, y1, z1, ..., x21, y21, z21]
    """
    landmarks = []
    # Raw coordinates (0-1), NO re-normalization like (x-0.5)*2
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    # Add hand type as prefix (1 for right, -1 for left)
    # inisiasi cllass.py logic: 'Right' -> 1, else -> -1
    hand_code = 1 if hand_type == "Right" else -1
    landmarks.insert(0, hand_code)
    
    return landmarks

# 4. Main Loop
cap = cv2.VideoCapture(0)
# Optional: Set resolution to match reference
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting video stream...")
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Mirror the frame immediately (like data collector)
    image = cv2.flip(image, 1)
    
    # Convert to RGB for MediaPipe
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    
    # Prepare for drawing
    image.flags.writeable = True
    
    prediction_text = "Waiting for hands..."
    
    # Initialize empty slots (64 zeros each), matching inisiasi cllass.py
    # 64 features = 1 hand_code + 21*3 coords
    left_hand_data = [0] * 64
    right_hand_data = [0] * 64
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get hand label (Right/Left)
            hand_label = handedness.classification[0].label
            
            # Extract features
            feats = extract_landmarks(hand_landmarks, hand_label)
            
            # Slot into correct array
            if hand_label == "Left":
                left_hand_data = feats
            else:
                right_hand_data = feats

            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        # Combine exactly as data collector: Left + Right
        combined_data = left_hand_data + right_hand_data # Length 128
        
        # Slice to 126 features to match Training logic
        target_size = 126
        features = np.array(combined_data[:target_size], dtype=np.float32)
        
        # Inference
        input_data = features.reshape(1, target_size)
        predictions = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)
        
        predicted_class = classes[predicted_index]
        conf_val = confidence * 100
        
        # Update UI
        if confidence > CONFIDENCE_THRESHOLD:
            color = (0, 255, 0) # Green
            prediction_text = f"Pred: {predicted_class} ({conf_val:.1f}%)"
        else:
            color = (0, 0, 255) # Red
            prediction_text = f"Low Conf: {predicted_class} ({conf_val:.1f}%)"
            
    # Display Text
    cv2.putText(image, prediction_text, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Bisindo Real-time Detection', image)
    
    if cv2.waitKey(5) & 0xFF == 27: # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
