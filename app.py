from flask import Flask
from flask_socketio import SocketIO, emit
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
import cv2  # Import OpenCV
from flask_cors import CORS
import eventlet  # To handle concurrent WebSocket connections

# Initialize Flask app and Flask-SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")  # Use eventlet for async

# Enable CORS for the app
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize HandDetector
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Debugging - log when client connects
@socketio.on('connect')
def test_connect():
    print('Client connected')

def detect_hands(image_data):
    try:
        # Convert image data to an OpenCV image
        np_img = np.array(Image.open(BytesIO(image_data)))

        # Convert the image from RGB to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # Detect hands in the image
        hands, _ = detector.findHands(img_bgr, flipType=False)

        keypoints = []
        if hands:
            for hand in hands:
                lmList = hand["lmList"]  # List of hand landmarks
                for idx, lm in enumerate(lmList):
                    keypoints.append(
                        {
                            "id": idx,
                            "x": lm[0],  # X-coordinate
                            "y": lm[1],  # Y-coordinate
                            "z": lm[2],  # Z-coordinate
                        }
                    )
        print("Hand keypoints detected:", keypoints)
        return keypoints

    except Exception as e:
        print("Error in hand detection:", str(e))
        raise e

@socketio.on("send_frame")
def handle_frame(frame_data):
    print("Received frame")
    try:
        # Decode the base64 image data
        img_data = base64.b64decode(frame_data.split(",")[1])
        print("Image data decoded successfully")

        # Detect hands from the image
        keypoints = detect_hands(img_data)
        print("Keypoints detected:", keypoints)

        # Emit the detected keypoints to the frontend
        emit("hand_keypoints", {"keypoints": keypoints})
        print("Keypoints emitted to frontend")

    except Exception as e:
        print("Error handling frame:", str(e))
        # Emit the error to frontend for debugging
        emit("error", {"error": str(e)})

@app.route("/")
def index():
    print("Hand Detection WebSocket Server is working")
    return "Hand Detection WebSocket Server"

if __name__ == "__main__":
    # Run the app with eventlet for better concurrent handling
    print("Starting the server...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_eventlet=True)
