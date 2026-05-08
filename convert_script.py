import sys
from types import ModuleType
# Mock the missing module
sys.modules['tensorflow_decision_forests'] = ModuleType('tensorflow_decision_forests')

import tensorflow as tf
import tensorflowjs as tfjs
import os

# Create output directory
if not os.path.exists('tfjs_model'):
    os.makedirs('tfjs_model')

print("Loading model...")
model = tf.keras.models.load_model('best_model.keras')
print("Model loaded.")

print("Converting model...")
try:
    tfjs.converters.save_keras_model(model, 'tfjs_model')
    print("Conversion complete.")
except Exception as e:
    print(f"Conversion failed: {e}")
