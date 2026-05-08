# 📁 File: data_collector_bisindo.py - VERSION FIXED
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from datetime import datetime
import pandas as pd

class BISINDODataCollector:
    def __init__(self, dataset_path="dataset_bisindo"):
        """
        Initialize BISINDO Data Collector
        - Supports both single-hand and double-hand gestures
        - Auto-detects hand presence
        """
        self.dataset_path = dataset_path
        
        # MediaPipe Setup - suppress warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Lower threshold for better detection
            min_tracking_confidence=0.5
        )
        
        # Dataset structure
        self.current_class = None
        self.sample_count = 0
        self.batch_size = 10
        self.batch_counter = 0
        
        # Create dataset directory if not exists
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "raw_data"), exist_ok=True)
        
        # Debug flag
        self.debug = True
        
    def initialize_class(self, class_name):
        """Initialize directory for a new class"""
        self.current_class = class_name
        
        # Create directories for all splits
        splits = ['train', 'val', 'test']
        for split in splits:
            class_path = os.path.join(self.dataset_path, split, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            # Create metadata file
            meta_file = os.path.join(class_path, "metadata.json")
            if not os.path.exists(meta_file):
                meta_data = {
                    "class_name": class_name,
                    "created": datetime.now().isoformat(),
                    "total_samples": 0,
                    "batches": [],
                    "hand_configurations": []
                }
                with open(meta_file, 'w') as f:
                    json.dump(meta_data, f, indent=4)
        
        print(f"✅ Kelas '{class_name}' diinisialisasi")
        self.sample_count = 0
        self.batch_counter = 0
        
    def extract_landmarks(self, hand_landmarks, hand_type="Right"):
        """
        Extract 21 landmarks for one hand
        Returns: [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
        """
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Add hand type as prefix (1 for right, -1 for left, 0 for missing)
        hand_code = 1 if hand_type == "Right" else -1
        landmarks.insert(0, hand_code)
        
        return landmarks
    
    def detect_and_extract(self, frame):
        """
        Process frame and extract landmarks for BOTH hands
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Initialize arrays
        left_hand_data = [0] * 64
        right_hand_data = [0] * 64
        detected_hands = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                  results.multi_handedness or []):
                # Get hand type
                hand_type = handedness.classification[0].label if handedness else "Unknown"
                
                # Extract landmarks
                landmarks = self.extract_landmarks(hand_landmarks, hand_type)
                
                # Store in appropriate array
                if hand_type == "Left":
                    left_hand_data = landmarks
                else:  # Right or Unknown
                    right_hand_data = landmarks
                
                detected_hands.append(hand_type)
                
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Combine data
        combined_data = left_hand_data + right_hand_data
        
        # Add metadata
        combined_data.extend([
            len(detected_hands),
            1 if "Left" in detected_hands else 0,
            1 if "Right" in detected_hands else 0
        ])
        
        if self.debug and detected_hands:
            print(f"DEBUG: Detected {len(detected_hands)} hand(s): {detected_hands}")
        
        return combined_data, detected_hands, frame
    
    def show_countdown(self, frame, countdown_time, message=""):
        """Show countdown on frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Countdown text
        if countdown_time > 0:
            countdown_text = f"{countdown_time}"
            text_size = cv2.getTextSize(countdown_text, font, 6, 10)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            cv2.putText(frame, countdown_text, (text_x, text_y), 
                       font, 6, (0, 255, 255), 10, cv2.LINE_AA)
        else:
            countdown_text = "GO!"
            text_size = cv2.getTextSize(countdown_text, font, 4, 10)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            cv2.putText(frame, countdown_text, (text_x, text_y), 
                       font, 4, (0, 255, 0), 10, cv2.LINE_AA)
        
        # Message
        if message:
            msg_size = cv2.getTextSize(message, font, 0.8, 2)[0]
            msg_x = (width - msg_size[0]) // 2
            cv2.putText(frame, message, (msg_x, 80), 
                       font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Instruction
        instruction = "Bersiap! Tetap posisikan tangan Anda"
        inst_size = cv2.getTextSize(instruction, font, 0.6, 2)[0]
        inst_x = (width - inst_size[0]) // 2
        cv2.putText(frame, instruction, (inst_x, height - 50), 
                   font, 0.6, (200, 200, 255), 2, cv2.LINE_AA)
        
        return frame
    
    def capture_batch(self, class_name, batch_num=1):
        """
        Simplified capture batch with better error handling
        """
        if self.current_class != class_name:
            self.initialize_class(class_name)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Tidak bisa membuka kamera!")
            return 0
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"\n📸 BATCH {batch_num} - Kelas: {class_name}")
        print("=" * 40)
        print("Petunjuk:")
        print("1. Pastikan tangan terlihat jelas di kamera")
        print("2. Tekan 's' untuk mulai (timer 3 detik)")
        print("3. Sistem akan otomatis mengambil 10 gambar")
        print("4. Tekan 'q' untuk keluar")
        print("=" * 40)
        
        # State variables
        state = "WAITING"  # WAITING, COUNTDOWN, CAPTURING, DONE
        countdown_start = 0
        capture_start = 0
        last_capture = 0
        capture_count = 0
        batch_samples = []
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        window_name = f'BISINDO - {class_name} (Batch {batch_num})'
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Gagal membaca frame!")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Process frame
            landmarks_data, detected_hands, processed_frame = self.detect_and_extract(display_frame)
            display_frame = processed_frame
            
            current_time = time.time()
            
            # State machine
            if state == "WAITING":
                # Display waiting screen
                cv2.putText(display_frame, f"Kelas: {class_name}", (10, 30), 
                           font, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Batch: {batch_num}", (10, 60), 
                           font, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Status: Menunggu...", (10, 90), 
                           font, 0.7, (255, 255, 0), 2)
                
                # Hand detection status
                if detected_hands:
                    status_text = f"✓ Tangan terdeteksi: {', '.join(detected_hands)}"
                    status_color = (0, 255, 0)
                else:
                    status_text = "⚠ Tidak ada tangan terdeteksi!"
                    status_color = (0, 0, 255)
                
                cv2.putText(display_frame, status_text, (10, 120), 
                           font, 0.6, status_color, 2)
                
                # Instructions
                cv2.putText(display_frame, "Tekan 's' untuk mulai", (10, 250), 
                           font, 0.6, (255, 255, 255), 1)
                cv2.putText(display_frame, "Tekan 'q' untuk keluar", (10, 280), 
                           font, 0.6, (255, 255, 255), 1)
                
                # Check for 's' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    if detected_hands:
                        state = "COUNTDOWN"
                        countdown_start = current_time
                        print("⏱️ Memulai countdown 3 detik...")
                        print("Bersiap! Tetap posisikan tangan Anda")
                    else:
                        print("⚠ Tidak ada tangan terdeteksi! Pastikan tangan terlihat jelas.")
                
                elif key == ord('q'):
                    print("⏹️ Menghentikan pengambilan data...")
                    break
            
            elif state == "COUNTDOWN":
                elapsed = current_time - countdown_start
                remaining = max(0, 3 - elapsed)
                
                if remaining > 0:
                    # Show countdown
                    countdown_frame = self.show_countdown(
                        display_frame, 
                        int(remaining) + 1,
                        f"Batch {batch_num} - {class_name}"
                    )
                    display_frame = countdown_frame
                else:
                    # Start capturing
                    state = "CAPTURING"
                    capture_start = current_time
                    last_capture = current_time
                    capture_count = 0
                    batch_samples = []
                    print("🚀 Memulai pengambilan gambar!")
            
            elif state == "CAPTURING":
                if capture_count < 10:
                    # Capture every 0.5 seconds
                    if current_time - last_capture >= 0.5:
                        # Save sample
                        sample_data = {
                            "class": class_name,
                            "batch": batch_num,
                            "sample_in_batch": capture_count + 1,
                            "timestamp": datetime.now().isoformat(),
                            "landmarks": landmarks_data,
                            "hand_count": len(detected_hands),
                            "hand_types": detected_hands
                        }
                        batch_samples.append(sample_data)
                        
                        print(f"📸 Gambar {capture_count + 1}/10 diambil")
                        last_capture = current_time
                        capture_count += 1
                    
                    # Display progress
                    progress = capture_count / 10
                    cv2.rectangle(display_frame, (50, 150), (600, 170), (100, 100, 100), 2)
                    cv2.rectangle(display_frame, (50, 150), 
                                 (50 + int(550 * progress), 170), (0, 255, 0), -1)
                    
                    cv2.putText(display_frame, f"Mengambil: {capture_count}/10 gambar", 
                               (10, 200), font, 0.6, (255, 255, 255), 2)
                    
                    # Next capture timer
                    if capture_count < 10:
                        next_time = 0.5 - (current_time - last_capture)
                        cv2.putText(display_frame, 
                                   f"Gambar berikutnya: {max(0, next_time):.1f}s", 
                                   (10, 230), font, 0.6, (255, 200, 100), 2)
                else:
                    # Capture complete
                    state = "DONE"
            
            elif state == "DONE":
                # Save the batch
                if batch_samples:
                    self.save_batch(batch_samples, batch_num)
                
                # Show completion message
                cv2.putText(display_frame, "BATCH SELESAI!", 
                           (200, 100), font, 1, (0, 255, 0), 3)
                cv2.putText(display_frame, f"{len(batch_samples)} sampel disimpan", 
                           (200, 140), font, 0.8, (0, 255, 255), 2)
                cv2.putText(display_frame, "Tekan sembarang tombol untuk keluar", 
                           (150, 180), font, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(2000)  # Show for 2 seconds
                break
            
            # Always show the frame
            cv2.imshow(window_name, display_frame)
            
            # Handle quit key in all states
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if state == "CAPTURING" and batch_samples:
                    self.save_batch(batch_samples, batch_num)
                print("⏹️ Menghentikan pengambilan data...")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyWindow(window_name)
        
        samples_collected = len(batch_samples)
        print(f"📊 Batch {batch_num} selesai: {samples_collected} sampel terkumpul")
        
        return samples_collected
    
    def save_batch(self, batch_samples, batch_num):
        """Save a batch of samples"""
        if not batch_samples:
            print("⚠ Tidak ada sampel untuk disimpan")
            return
        
        class_name = self.current_class
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"💾 Menyimpan {len(batch_samples)} sampel untuk kelas {class_name}...")
        
        try:
            # Save each sample
            for i, sample in enumerate(batch_samples):
                # Determine split
                if self.sample_count % 10 < 8:
                    split = "train"
                elif self.sample_count % 10 == 8:
                    split = "val"
                else:
                    split = "test"
                
                # Create filename
                filename = f"{class_name}_b{batch_num:02d}_s{i+1:02d}_{timestamp}.npy"
                filepath = os.path.join(self.dataset_path, split, class_name, filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save as numpy array
                landmarks_array = np.array(sample["landmarks"], dtype=np.float32)
                np.save(filepath, landmarks_array)
                
                # Print debug info for first sample
                if i == 0:
                    print(f"  DEBUG: Landmarks shape: {landmarks_array.shape}")
                    print(f"  DEBUG: Saved to: {filepath}")
                
                # Update metadata
                self.update_metadata(class_name, split, sample)
                
                self.sample_count += 1
            
            print(f"✅ Batch {batch_num} berhasil disimpan")
            
            # Save CSV backup
            self.save_batch_csv(batch_samples, batch_num, timestamp)
            
        except Exception as e:
            print(f"❌ Error menyimpan batch: {e}")
            import traceback
            traceback.print_exc()
    
    def update_metadata(self, class_name, split, sample):
        """Update metadata JSON file"""
        meta_file = os.path.join(self.dataset_path, split, class_name, "metadata.json")
        
        # Create metadata if doesn't exist
        if not os.path.exists(meta_file):
            meta_data = {
                "class_name": class_name,
                "created": datetime.now().isoformat(),
                "total_samples": 0,
                "batches": [],
                "hand_configurations": []
            }
        else:
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
        
        # Update
        meta_data["total_samples"] = meta_data.get("total_samples", 0) + 1
        
        batch_num = sample["batch"]
        if batch_num not in meta_data["batches"]:
            meta_data["batches"].append(batch_num)
        
        if sample['hand_types']:
            hand_config = f"{sample['hand_count']}_hand_{'_'.join(sample['hand_types'])}"
        else:
            hand_config = "0_hand"
            
        if hand_config not in meta_data["hand_configurations"]:
            meta_data["hand_configurations"].append(hand_config)
        
        # Save
        with open(meta_file, 'w') as f:
            json.dump(meta_data, f, indent=4)
    
    def save_batch_csv(self, batch_samples, batch_num, timestamp):
        """Save batch as CSV"""
        try:
            csv_file = os.path.join(self.dataset_path, "raw_data", 
                                   f"{self.current_class}_batch{batch_num}_{timestamp}.csv")
            
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            
            # Prepare data
            data = []
            for sample in batch_samples:
                row = {
                    "class": sample["class"],
                    "batch": sample["batch"],
                    "sample": sample["sample_in_batch"],
                    "hand_count": sample["hand_count"],
                    "hand_types": ",".join(sample["hand_types"]) if sample["hand_types"] else "none",
                    "timestamp": sample["timestamp"]
                }
                
                # Add first few landmarks for inspection
                landmarks = sample["landmarks"]
                for j in range(min(5, len(landmarks))):
                    row[f"lm_{j}"] = landmarks[j]
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
            print(f"📄 CSV backup: {csv_file}")
            
        except Exception as e:
            print(f"⚠ Gagal menyimpan CSV: {e}")


class DatasetInitializer:
    @staticmethod
    def initialize_dataset_structure(classes, dataset_path="dataset_bisindo"):
        """Initialize the complete dataset structure"""
        print("📁 Membuat struktur dataset...")
        
        # Create main directories
        directories = [
            dataset_path,
            os.path.join(dataset_path, "raw_data"),
            os.path.join(dataset_path, "models"),
            os.path.join(dataset_path, "logs")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"  📂 {directory}")
        
        # Create class directories
        splits = ['train', 'val', 'test']
        for class_name in classes:
            for split in splits:
                class_path = os.path.join(dataset_path, split, class_name)
                os.makedirs(class_path, exist_ok=True)
        
        # Create info file
        dataset_info = {
            "dataset_name": "BISINDO_Dataset",
            "created": datetime.now().isoformat(),
            "classes": classes,
            "num_classes": len(classes),
            "description": "Dataset Bahasa Isyarat Indonesia"
        }
        
        info_file = os.path.join(dataset_path, "dataset_info.json")
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        print(f"\n✅ Struktur dataset siap di: {dataset_path}")
        return dataset_info


def test_collection():
    """Simple test function"""
    print("=" * 60)
    print("TEST BISINDO DATA COLLECTOR")
    print("=" * 60)
    
    # Initialize
    initializer = DatasetInitializer()
    initializer.initialize_dataset_structure(['TEST'])
    
    collector = BISINDODataCollector(dataset_path="dataset_bisindo")
    
    print("\n🎯 TEST INSTRUKSI:")
    print("1. Buka kamera dan pastikan tangan terlihat")
    print("2. Tunggu sampai status '✓ Tangan terdeteksi' muncul")
    print("3. Tekan 's' untuk mulai")
    print("4. Tunggu countdown 3 detik")
    print("5. Sistem akan otomatis capture 10 gambar")
    print("6. Periksa folder 'dataset_bisindo' untuk hasil")
    
    input("\nTekan Enter untuk mulai test...")
    
    # Run test
    samples = collector.capture_batch('TEST', 1)
    
    print(f"\n{'='*60}")
    if samples > 0:
        print(f"✅ TEST BERHASIL! {samples} sampel terkumpul")
        
        # Check saved files
        train_path = "dataset_bisindo/train/TEST"
        if os.path.exists(train_path):
            npy_files = [f for f in os.listdir(train_path) if f.endswith('.npy')]
            print(f"📁 File .npy yang tersimpan: {len(npy_files)}")
            if npy_files:
                print(f"  Contoh: {npy_files[0]}")
    else:
        print("❌ TEST GAGAL! Tidak ada sampel terkumpul")
        print("   Pastikan:")
        print("   1. Kamera berfungsi")
        print("   2. Tangan terlihat jelas")
        print("   3. Pencahayaan cukup")
        print("   4. Tidak menutupi kamera")
    
    print("=" * 60)


def main_collection():
    """Main collection function"""
    print("=" * 60)
    print("BISINDO DATA COLLECTION - MAIN")
    print("=" * 60)
    
    # Get classes
    print("\nPilih kelas:")
    print("1. Huruf A (testing)")
    print("2. A, - Z (26 huruf)")
    print("3. Custom (input manual)")
    
    choice = input("\nPilihan (1/2/3): ")
    
    if choice == "1":
        classes = ['A']
    elif choice == "2":
        classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

    else:
        custom = input("Masukkan huruf (pisah koma): ")
        classes = [c.strip().upper() for c in custom.split(',')]
    
    # Get batches
    try:
        batches = int(input("Jumlah batch per kelas (default 3): ") or "3")
    except:
        batches = 3
    
    print(f"\n📋 Rencana:")
    print(f"  Kelas: {', '.join(classes)}")
    print(f"  Batch per kelas: {batches}")
    print(f"  Target sampel: {len(classes) * batches * 10}")
    
    # Initialize
    initializer = DatasetInitializer()
    initializer.initialize_dataset_structure(classes)
    
    collector = BISINDODataCollector(dataset_path="dataset_bisindo")
    
    input("\n📝 Tekan Enter untuk mulai...")
    
    # Collect for each class
    total_samples = 0
    for class_idx, class_name in enumerate(classes):
        print(f"\n{'='*60}")
        print(f"🎬 [{class_idx+1}/{len(classes)}] Koleksi: {class_name}")
        print(f"{'='*60}")
        
        print(f"\nPosisikan tangan untuk huruf '{class_name}'")
        print("Pastikan:")
        print("  1. Tangan terlihat jelas")
        print("  2. Pencahayaan cukup")
        print("  3. Background tidak terlalu ramai")
        input("Tekan Enter ketika siap...")
        
        class_samples = 0
        for batch_num in range(1, batches + 1):
            print(f"\n🔄 Batch {batch_num}/{batches}")
            print("-" * 30)
            
            samples = collector.capture_batch(class_name, batch_num)
            class_samples += samples
            
            print(f"📊 Batch ini: {samples} sampel")
            print(f"📊 Total {class_name}: {class_samples}/{batches * 10}")
            
            if batch_num < batches and samples > 0:
                print("⏸️  Istirahat 1 detik...")
                time.sleep(1)
        
        total_samples += class_samples
        print(f"\n✅ {class_name} selesai: {class_samples} sampel")
        
        if class_idx < len(classes) - 1:
            next_class = classes[class_idx + 1]
            cont = input(f"\nLanjut ke '{next_class}'? (y/n): ")
            if cont.lower() != 'y':
                print("⏹️ Dihentikan")
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 KOLEKSI SELESAI!")
    print(f"{'='*60}")
    print(f"📊 Total sampel: {total_samples}")
    print(f"📁 Dataset: dataset_bisindo/")
    
    # Show file count
    train_path = "dataset_bisindo/train"
    if os.path.exists(train_path):
        for class_dir in os.listdir(train_path):
            if os.path.isdir(os.path.join(train_path, class_dir)):
                files = [f for f in os.listdir(os.path.join(train_path, class_dir)) 
                        if f.endswith('.npy')]
                if files:
                    print(f"  {class_dir}: {len(files)} file")
    
    print("\n✅ Data siap untuk training!")


if __name__ == "__main__":
    print("BISINDO DATA COLLECTOR")
    print("=" * 40)
    print("1. Test Mode (rekomendasi pertama)")
    print("2. Main Collection")
    print("3. Exit")
    
    option = input("\nPilih (1/2/3): ")
    
    if option == "1":
        test_collection()
    elif option == "2":
        main_collection()
    else:
        print("👋 Sampai jumpa!")