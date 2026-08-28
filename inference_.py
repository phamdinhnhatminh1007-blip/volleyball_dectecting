import streamlit as st
import cv2
import numpy as np
import requests

st.title("🏐 Volleyball: Ball & Player Detection")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

uploaded_file = st.file_uploader("Uploading image", type=["jpg", "jpeg", "png"])

# Chỉ chạy logic khi đã có ảnh
if uploaded_file is not None:
    # 1. Đọc và giải mã ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # 2. Lấy API Key và chuẩn bị request
    # Đổi tên biến cho đúng ngữ cảnh (trong file secrets.toml nhớ đặt là ROBOFLOW_API_KEY)
    YOUR_API_KEY = st.secrets["ROBOFLOW_API_KEY"] 
    url = f"https://detect.roboflow.com/volleyball-video-jl3zk/1?api_key={YOUR_API_KEY}"
    
    # Nén lại để gửi đi
    _, img_encoded = cv2.imencode('.jpg', frame)
    res = requests.post(url, files={"file": img_encoded.tobytes()})
    
    # 3. Xử lý kết quả trả về
    if res.status_code == 200:
        preds = res.json().get("predictions", [])
        ball_count = 0
        player_count = 0

        # Xử lý logic & Vẽ khung
        for pred in preds:
            if pred["confidence"] < conf_thresh:
                continue
            
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            cls = pred["class"].lower() # Đưa về chữ thường: "ball" hoặc "player"

            # Logic gán nhãn
            label = "Ball" if cls == "ball" else "Player"
            color = (0, 255, 0) if label == "Ball" else (0, 0, 255)

            if label == "Ball":
                ball_count += 1
            else:
                player_count += 1

            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 3)
            # Tùy chọn thêm: Viết chữ lên khung cho rõ
            cv2.putText(frame, label, (x - w//2, y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 4. Hiển thị Dashboard chỉ số & Ảnh kết quả
        col1, col2 = st.columns(2)
        col1.metric("Player", player_count)
        col2.metric("Ball", ball_count)

        # Chuyển BGR -> RGB để Streamlit hiển thị đúng màu
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Kết quả nhận diện", use_container_width=True)
        
    else:
        st.error(f"Lỗi API: {res.status_code} - {res.text}")