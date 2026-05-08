import cv2
import mediapipe as mp
import numpy as np
import torch
from train_mlp_torch import BisindoMLP, INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, MODEL_SAVE_PATH

def realtime_detection():
    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    # Ensure this matches training script's input size
    # Training script slices [:126], so we must do the same.
    model = BisindoMLP(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return

    model.eval()

    # 2. Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Classes A-Z
    classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]

    # 3. Webcam Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Use 640x480 to match potential data collection setup, though not strictly required
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting webcam... Press 'q' to quit.")

    # Helper function to match inisiasi cllass.py
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

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2) as hands:
        
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
                
                # Slice to INPUT_SIZE (126) to match Training logic
                features = np.array(combined_data[:INPUT_SIZE], dtype=np.float32)
                
                # Prediction
                input_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, 1)
                    conf, pred_idx = torch.max(probs, 1)
                    
                    pred_char = classes[pred_idx.item()]
                    conf_val = conf.item() * 100
                    
                    prediction_text = f"Pred: {pred_char} ({conf_val:.1f}%)"

            # Display Text
            cv2.putText(image, prediction_text, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Bisindo Real-time Detection', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_detection()
