import tensorflow as tf
import numpy as np
import os
import glob
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (safe for server/headless)
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

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
    model.save('bisindo_mlp_model.h5')  # Legacy format
    print("Model saved to bisindo_mlp_model.keras and bisindo_mlp_model.h5")

    # Save classes
    np.save('classes.npy', classes)
    print("Classes saved to classes.npy")

    # --- Evaluation & Visualization ---
    evaluate_and_save(model, history, X_val, y_val_encoded, classes)


def evaluate_and_save(model, history, X_val, y_val_encoded, classes):
    """Generate and save evaluation plots to 'hasil_evaluasi' folder."""

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("hasil_evaluasi", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving evaluation results to: {output_dir}")

    # ── 1. Accuracy & Loss Curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight='bold')

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='#2196F3', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='#FF5722', linewidth=2, linestyle='--')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss', color='#4CAF50', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss',   color='#9C27B0', linewidth=2, linestyle='--')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    acc_loss_path = os.path.join(output_dir, "accuracy_loss_curves.png")
    plt.savefig(acc_loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Accuracy & Loss curves saved: {acc_loss_path}")

    # ── 2. Confusion Matrix ────────────────────────────────────────────────────
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_val_encoded, y_pred)
    n_classes = len(classes)
    fig_size = max(12, n_classes * 0.5)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Confusion matrix saved  : {cm_path}")

    # ── 3. Classification Report (text + bar chart) ────────────────────────────
    report_txt = classification_report(y_val_encoded, y_pred, target_names=classes)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report_txt)
    print(f"  [OK] Classification report   : {report_path}")
    print("\n" + report_txt)

    # Per-class F1-score bar chart
    report_dict = classification_report(
        y_val_encoded, y_pred, target_names=classes, output_dict=True
    )
    f1_scores  = [report_dict[c]['f1-score']  for c in classes]
    precisions = [report_dict[c]['precision'] for c in classes]
    recalls    = [report_dict[c]['recall']    for c in classes]

    x = np.arange(len(classes))
    width = 0.28
    fig, ax = plt.subplots(figsize=(max(14, n_classes * 0.7), 6))
    ax.bar(x - width, precisions, width, label='Precision', color='#2196F3', alpha=0.85)
    ax.bar(x,         recalls,    width, label='Recall',    color='#4CAF50', alpha=0.85)
    ax.bar(x + width, f1_scores,  width, label='F1-Score',  color='#FF5722', alpha=0.85)
    ax.set_title('Per-Class Precision / Recall / F1-Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "classification_report_chart.png")
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Classification bar chart: {bar_path}")

    print(f"\n✅ Semua hasil evaluasi tersimpan di folder: {output_dir}")


if __name__ == "__main__":
    main()
