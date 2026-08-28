# 🏐 Volleyball: Ball & Player Detection

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://volleyballdectecting.streamlit.app/)

A Computer Vision web application built with Streamlit and OpenCV that automatically detects volleyball players and the ball in uploaded images. The detection is powered by a custom object detection model deployed via the Roboflow API.

## ✨ Features

* **Instant Object Detection:** Upload any volleyball match image and instantly get bounding boxes drawn around players (Red) and the ball (Green).
* **Adjustable Confidence Threshold:** Users can easily tweak the model's confidence threshold using a sidebar slider to filter out weak or incorrect predictions.
* **Live Dashboard:** Displays a real-time count metric of the total players and balls detected in the frame.
* **In-Memory Processing:** All image decoding, encoding, and drawing operations are handled entirely in RAM (using OpenCV and NumPy) without saving files to the local disk, ensuring zero-latency performance.

## 🛠️ Tech Stack

* **Frontend & Deployment:** [Streamlit](https://streamlit.io/)
* **Computer Vision:** [OpenCV](https://opencv.org/) (`opencv-python-headless`)
* **Matrix Operations:** [NumPy](https://numpy.org/)
* **Model Inference:** [Roboflow API](https://roboflow.com/)
* **Network Requests:** `requests`

## 🚀 How to Run Locally

Follow these steps to set up and run the project on your personal machine.

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/volleyball_dectecting.git](https://github.com/your-username/volleyball_dectecting.git)
cd volleyball_dectecting
