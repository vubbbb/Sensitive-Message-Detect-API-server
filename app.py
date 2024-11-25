from flask import Flask, request, jsonify
import numpy as np
from scipy.stats import mode
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re

app = Flask(__name__)

# Tải mô hình và vectorizer đã lưu
with open("./knn_model.pkl", "rb") as knn_file:
    knn_model = pickle.load(knn_file)

with open("./lr_model.pkl", "rb") as lr_file:
    lr_model = pickle.load(lr_file)

with open("./nb_model.pkl", "rb") as nb_file:
    nb_model = pickle.load(nb_file)

# Load dữ liệu
data = pd.read_csv('./sensitive_messages.csv')
X = data['text'].tolist()
y = data['label'].tolist()

# Làm sạch dữ liệu
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Loại bỏ ký tự đặc biệt
    return text

X_cleaned = [clean_text(x) for x in X]

# Tách tập train và test
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)

# Trích đặc trưng bằng TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(X_train)

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

        # Dự đoán nhãn từ các mô hình
        knn_prediction = knn_model.predict(text_vectorized)[0]
        lr_prediction = lr_model.predict(text_vectorized)[0]
        nb_prediction = nb_model.predict(text_vectorized)[0]
        
        # Kết hợp các dự đoán từ các mô hình (Lấy giá trị mode)
        combined_predictions = np.array([knn_prediction, lr_prediction, nb_prediction])
        final_predictions, _ = mode(combined_predictions, axis=0)

        # Trả kết quả dự đoán
        return jsonify({"text": text, "prediction": final_predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
