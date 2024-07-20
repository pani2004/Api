import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import base64

app = Flask(__name__)
CORS(app)

model_path = os.path.join('models', 'xray_v3.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' does not exist.")
model = load_model(model_path)

labels = {0: 'Normal', 1: 'Tuberculosis', 2: 'Pneumonia'}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400

    try:
        predictions = []
        files = request.files.getlist('image')

        for file in files:
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            image_stream = BytesIO(file.read())
            img = load_img(image_stream, target_size=(224, 224), color_mode='rgb')  
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = labels[predicted_class]

            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((10, 10), predicted_label, font=font, fill=(255, 0, 0))

            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)

            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            predictions.append({
                'filename': file.filename,
                'label': predicted_label,
                'image': img_base64
            })

        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



