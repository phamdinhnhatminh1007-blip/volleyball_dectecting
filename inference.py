import streamlit as st
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import base64
import tempfile
import os

# --- THIẾT LẬP GIAO DIỆN TRANG WEB ---
st.set_page_config(page_title="Nhận diện Bóng chuyền", layout="centered")
st.title("🏐 Ứng dụng Nhận diện Bóng chuyền")
st.write("Tải một bức ảnh lên để mô hình AI phân tích và trả về kết quả.")

# --- CẤU HÌNH API ROBOFLOW ---
API_KEY = "FBrKWTJy48jWcHmng3uC"
WORKSPACE_NAME = "nh-nht-minh-phm-s-workspace"
WORKFLOW_ID = "volleyball-video-vvolleyball-video-jl3zk-1-yolov8n-t1-logic-2"

# Sử dụng @st.cache_resource để không phải kết nối lại API mỗi lần bấm nút
@st.cache_resource
def get_client():
    # Sử dụng URL serverless của Roboflow để chạy online thay vì localhost
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com", 
        api_key=API_KEY
    ).configure(InferenceConfiguration(
        api_key_transport="header" 
    ))

client = get_client()

# --- KHU VỰC TẢI ẢNH LÊN ---
uploaded_file = st.file_uploader("Chọn ảnh từ máy của bạn (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh gốc vừa tải lên
    st.image(uploaded_file, caption="Ảnh gốc", use_container_width=True)
    
    # Nút bấm chạy AI
    if st.button("🚀 Chạy Nhận Diện", type="primary"):
        with st.spinner("Đang gửi dữ liệu lên AI, vui lòng đợi..."):
            try:
                # Tạo file tạm thời để lưu ảnh vừa tải lên (vì workflow cần đường dẫn file)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Chạy Workflow dự đoán
                result = client.run_workflow(
                    workspace_name=WORKSPACE_NAME,
                    workflow_id=WORKFLOW_ID,
                    images={
                        "image": tmp_path
                    },
                    use_cache=True 
                )

                # Giải mã bức ảnh kết quả dạng base64
                base64_string = result[0]['bounding_box_visualization_output']
                img_bytes = base64.b64decode(base64_string)

                # Hiển thị kết quả lên màn hình web
                st.success("✅ Phân tích thành công!")
                st.image(img_bytes, caption="Kết quả nhận diện", use_container_width=True)

                # Cung cấp nút tải ảnh kết quả về
                st.download_button(
                    label="⬇️ Tải ảnh kết quả về máy",
                    data=img_bytes,
                    file_name="ket_qua_nhan_dien.jpg",
                    mime="image/jpeg"
                )

                # Xóa file tạm để giải phóng bộ nhớ
                os.remove(tmp_path)

            except Exception as e:
                st.error(f"❌ Có lỗi xảy ra trong quá trình nhận diện: {e}")
