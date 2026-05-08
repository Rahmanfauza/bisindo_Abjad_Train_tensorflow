import tensorflow as tf
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datetime

# 1. Configuration
DATASET_PATH = "dataset_bisindo"
INPUT_SHAPE = (126,) # We will slice the first 126 features (21 landmarks * 3 coords * 2 hands)
NUM_CLASSES = 26
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# 2. Data Loading Function
def load_data(split_name):
    """Loads data from the specified split directory (train/val/test)."""
    data = []
    labels = []
    
    split_dir = os.path.join(DATASET_PATH, split_name)
    if not os.path.exists(split_dir):
        print(f"Warning: Directory {split_dir} not found.")
        return np.array([]), np.array([])

    print(f"Loading {split_name} data...")
    
    # Iterate over class folders
    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"  Processing class: {class_name}")
        
        # Iterate over .npy files
        for file_name in glob.glob(os.path.join(class_dir, "*.npy")):
            try:
                # Load numpy array
                sample = np.load(file_name)
                
                # Preprocessing: Slice to first 126 features if larger
                if sample.shape[0] > 126:
                    sample = sample[:126]
                elif sample.shape[0] < 126:
                     # Pad with zeros if smaller (unlikely based on inspection)
                    sample = np.pad(sample, (0, 126 - sample.shape[0]), 'constant')

                data.append(sample)
                labels.append(class_name)
            except Exception as e:
                print(f"    Error loading {file_name}: {e}")

    return np.array(data), np.array(labels)

# 3. Main Training Script
def main():
    # --- Load Data ---
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")
    # If val is empty, split train data
    if len(X_val) == 0 and len(X_train) > 0:
         print("Validation set empty, splitting training data...")
         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    if len(X_train) == 0:
        print("Error: No training data found!")
        return

    # Encode Labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    classes = label_encoder.classes_
    print(f"Classes: {classes}")
    
    # Convert to One-Hot
    y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=NUM_CLASSES)
    y_val_onehot = tf.keras.utils.to_categorical(y_val_encoded, num_classes=NUM_CLASSES)

    # Create tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot))
    train_ds = train_ds.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_onehot))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- Build Model ---
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=INPUT_SHAPE),
        
        # Layer 1
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Layer 2
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Layer 3
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Output Layer
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # --- Callbacks ---
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True),
        tensorboard_callback
    ]

    # --- Train ---
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # --- Save Model ---
    model.save('bisindo_mlp_model.keras')
    model.save('bisindo_mlp_model.h5') # Legacy format
    print("Model saved to bisindo_mlp_model.keras and bisindo_mlp_model.h5")

    # Save classes
    np.save('classes.npy', classes)
    print("Classes saved to classes.npy")

if __name__ == "__main__":
    main()
