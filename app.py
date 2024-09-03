from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

# Load or define your model here
def build_classification_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # Update to match number of classes
    ])
    return model

# Initialize and compile the model
classification_model = build_classification_model()
classification_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load pre-trained weights if applicable
# classification_model.load_weights('path_to_weights.h5')

# Function to suggest medication based on the category
def suggest_medication(category):
    if category == 0:
        return "Use over-the-counter creams and monitor the symptom."
    elif category == 1:
        return "Consult a dermatologist for a prescription medication."
    elif category == 2:
        return "Immediate medical attention required. Possible surgical intervention."
    else:
        return "Invalid category"

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a POST request at /predict")
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        data = request.json
        image_base64 = data['image']

        # Decode Base64 image data
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((128, 128))  # Resize image to match model input
        image = np.array(image)  # Convert to NumPy array

        if image.shape != (128, 128, 3):
            return jsonify({'error': 'Invalid image shape'}), 400
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict the category
        predictions = classification_model.predict(image)
        predicted_category = np.argmax(predictions)

        # Get the medication suggestion
        medication_suggestion = suggest_medication(predicted_category)

        return jsonify({
            'predicted_category': int(predicted_category),  # Convert to int for JSON serialization
            'medication_suggestion': medication_suggestion
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
