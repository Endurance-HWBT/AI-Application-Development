# app.py - Save this file
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from PIL import Image
import io
import google.generativeai as genai
import webbrowser
from threading import Timer

app = Flask(__name__)
CORS(app)

# CONFIG
PROJECT_DIR = Path("runs")
GEMINI_API_KEY = ""#removed for safety
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# Load YOLO model
def load_model():
    run_folders = [d for d in PROJECT_DIR.iterdir() if d.is_dir()]
    latest_run = max(run_folders, key=lambda p: p.stat().st_mtime)
    weights = latest_run / "weights" / "best.pt"
    if not weights.exists():
        weights = latest_run / "weights" / "last.pt"
    return YOLO(weights)


print("Loading model...")
model = load_model()
print("✅ Model loaded!")


# Serve the HTML file
@app.route('/')
def home():
    return send_file('final_frontend.html')


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        img_array = np.array(img)

        # YOLO prediction
        results = model.predict(img_array, verbose=False, imgsz=224, conf=0.5)

        if results:
            r = results[0]
            label = r.names[r.probs.top1]
            conf = r.probs.top1conf.item()

            # Gemini advice
            prompt = f"Apple leaf disease: {label}. Give brief treatment in bullets: causes, organic/chemical solutions, prevention. Be concise."
            try:
                response = gemini_model.generate_content(prompt)
                advice = response.text
            except Exception as e:
                advice = f"⚠️ AI advice unavailable: {str(e)}"

            return jsonify({
                'success': True,
                'disease': label,
                'confidence': f'{conf * 100:.1f}',
                'advice': advice
            })

        return jsonify({'success': False, 'error': 'No disease detected'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# Auto-open browser
def open_browser():
    webbrowser.open('http://localhost:5000')


if __name__ == '__main__':
    print("\n🚀 Starting server at http://localhost:5000")
    print("📂 Make sure index.html is in the same folder!\n")
    Timer(1, open_browser).start()  # Auto-open browser after 1 sec
    app.run(debug=True, port=5000, use_reloader=False)