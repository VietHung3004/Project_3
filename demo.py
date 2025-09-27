import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd

# --- Hàm làm sạch văn bản ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^0-9a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
                  r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ"
                  r"ỳýỵỷỹđ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Các file mô hình ---
models_info = {
    "LSTM": "best_LSTM.keras",
    "GRU": "best_BiGru.keras",
    "BiLSTM": "best_BiLSTM.keras"
}

tokenizer_path = "tokenizer25_9_2025.pkl"
label_encoder_path = "label_encoder.joblib"

# --- Load tokenizer ---
try:
    tokenizer = joblib.load(tokenizer_path)
except FileNotFoundError:
    st.error(f"Không tìm thấy file tokenizer tại: {tokenizer_path}")
    st.stop()

# --- Load label encoder ---
try:
    le = joblib.load(label_encoder_path)
except FileNotFoundError:
    st.error(f"Không tìm thấy file label encoder tại: {label_encoder_path}")
    st.stop()

# --- Giao diện chọn mô hình ---
st.set_page_config(page_title="Dự đoán chủ đề văn bản", layout="wide")
st.title("📄 Dự đoán chủ đề văn bản tiếng Việt")
st.write("Nhập văn bản và chọn mô hình để dự đoán chủ đề.")

selected_model_name = st.radio("Chọn mô hình:", list(models_info.keys()))
model_path = models_info[selected_model_name]

# --- Load model ---
try:
    model = tf.keras.models.load_model(model_path)
except FileNotFoundError:
    st.error(f"Không tìm thấy file mô hình tại: {model_path}")
    st.stop()

# --- Hàm dự đoán ---
def predict_text(text, max_len=650):
    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)
    pred_label = np.argmax(pred, axis=1)[0]
    label = le.inverse_transform([pred_label])[0]
    probabilities = pred[0]
    return label, probabilities

# --- Nhập văn bản ---
text_input = st.text_area("✍️ Nhập văn bản:", height=200)

# --- Nút dự đoán ---
if st.button("🔍 Dự đoán"):
    if text_input.strip() == "":
        st.warning("⚠️ Vui lòng nhập văn bản để dự đoán!")
    else:
        predicted_topic, probabilities = predict_text(text_input)
        
        st.success(f"✅ **Chủ đề dự đoán (bằng {selected_model_name})**: {predicted_topic}")

        # Hiển thị xác suất từng lớp
        prob_df = pd.DataFrame({
            "Chủ đề": le.classes_,
            "Xác suất": probabilities
        }).sort_values(by="Xác suất", ascending=False)

        st.subheader("📊 Xác suất từng chủ đề")
        for i, row in prob_df.iterrows():
            emoji = "🏆" if row['Chủ đề'] == predicted_topic else "📌"
            st.markdown(f"{emoji} **{row['Chủ đề']}**: {row['Xác suất']*100:.2f}%")
