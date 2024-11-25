from flask import Flask, request, jsonify
import joblib
import numpy as np
from scipy.stats import mode

app = Flask(__name__)

# Tải mô hình và vectorizer đã lưu
knn_model = joblib.load("./knn_model.pkl")
lr_model = joblib.load("./lr_model.pkl")
nb_model = joblib.load("./nb_model.pkl")
vectorizer = joblib.load("./vectorizer.pkl")

@app.route("/")
def home():
    return "Welcome to the AI API Server!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Lấy dữ liệu từ request
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        # Văn bản đầu vào
        text = data["text"]

        # Chuyển đổi văn bản sang đặc trưng TF-IDF
        text_vectorized = vectorizer.transform([text])

        # Dự đoán nhãn
        knn_prediction = knn_model.predict(text_vectorized)[0]
        lr_prediction = lr_model.predict(text_vectorized)[0]
        nb_prediction = nb_model.predict(text_vectorized)[0]
        
        combined_predictions = np.array([knn_prediction, lr_prediction, nb_prediction])
        final_predictions, _ = mode(combined_predictions, axis=0)

        # Trả kết quả
        return jsonify({"text": text, "prediction": final_predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
