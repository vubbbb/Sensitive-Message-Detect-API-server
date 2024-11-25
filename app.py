from flask import Flask, request, jsonify
import numpy as np
from scipy.stats import mode
import pickle

app = Flask(__name__)

# Tải mô hình và vectorizer đã lưu
with open("./knn_model.pkl", "rb") as knn_file:
    knn_model = pickle.load(knn_file)

with open("./lr_model.pkl", "rb") as lr_file:
    lr_model = pickle.load(lr_file)

with open("./nb_model.pkl", "rb") as nb_file:
    nb_model = pickle.load(nb_file)

with open("./vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

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
