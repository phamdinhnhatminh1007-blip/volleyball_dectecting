import cv2
import requests
import sys

# =========================================
# CONFIGURATION
# =========================================
API_KEY = "FBrKWTJy48jWcHmng3uC"
PROJECT_ID = "volleyball-video-jl3zk"
MODEL_VERSION = "1"

# Roboflow REST API endpoint for inference
INFERENCE_URL = f"https://detect.roboflow.com/{PROJECT_ID}/{MODEL_VERSION}?api_key={API_KEY}"

# Input and output videos
INPUT_VIDEO = "slipsv2.mp4"  # Path to input video
OUTPUT_VIDEO = "fall_output_annotatedv8.mp4" # Path for saving annotated output

# Confidence threshold
CONF_THRESHOLD = 0.2

# =========================================
# FUNCTION TO RUN INFERENCE
# =========================================
def infer_frame(frame):
    # Encode frame as JPG before sending to Roboflow API
    _, img_encoded = cv2.imencode('.jpg', frame)

    # Send frame for prediction
    response = requests.post(
        INFERENCE_URL,
        files={"file": img_encoded.tobytes()},
        data={"name": "video_frame"}
    )
    # If successful, return predictions as JSON
    if response.status_code == 200:
        return response.json()
    else:
        # Print error if API call fails
        print("Error:", response.text)
        return None

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    # Kiểm tra xem có mở được video không
    if not cap.isOpened():
        print(f"Error: Không thể mở video đầu vào: {INPUT_VIDEO}")
        sys.exit()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:  # Exit loop if no more frames
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        predictions = infer_frame(frame)

        if predictions and "predictions" in predictions:
            for pred in predictions["predictions"]:
                confidence = pred["confidence"]

                # Skip low-confidence detections
                if confidence < CONF_THRESHOLD:
                    continue
                
                # Extract bounding box center coordinates and size
                x, y = int(pred["x"]), int(pred["y"])
                w, h = int(pred["width"]), int(pred["height"])
                class_name = pred["class"].lower()

                # ===== FLIP THE LABELS HERE =====
                if class_name == "fall":
                    display_label = "stand"
                    color = (0, 255, 0)  # green for "stand"
                elif class_name == "stand":
                    display_label = "fall"
                    color = (0, 0, 255)  # red for "fall"
                else:
                    display_label = class_name
                    color = (255, 255, 0)  # yellow for any unknown class
                
                # Draw bounding box around detected object
                cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 5)

                # Draw label text above the bounding box
                label = f"{display_label} ({confidence:.2f})"
                cv2.putText(frame, label, (x - w//2, y - h//2 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 5)

        # Write annotated frame to output video
        out.write(frame)

        # Show frame in a window (press 'q' to stop early)
        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user")
            break

    # Phần dọn dẹp tài nguyên đã được đưa vào đúng thụt lề bên trong hàm main()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Processing complete. Annotated video saved as", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()