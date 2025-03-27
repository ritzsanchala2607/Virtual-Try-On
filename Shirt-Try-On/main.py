import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()       
mp_drawing = mp.solutions.drawing_utils

# Load T-shirt image
tshirt_img = cv2.imread("Resources/Shirts/1.png", cv2.IMREAD_UNCHANGED)

# Start webcam capture
cap = cv2.VideoCapture(0)

# Set fullscreen window
cv2.namedWindow("Virtual Try-On", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Virtual Try-On", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe Pose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get shoulder coordinates
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Convert to pixel values
        h, w, _ = frame.shape
        left_shoulder = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder = (int(right_shoulder.x * w), int(right_shoulder.y * h))

        # Calculate T-shirt width and height
        tshirt_width = abs(right_shoulder[0] - left_shoulder[0]) * 2
        tshirt_height = int(tshirt_width * (tshirt_img.shape[0] / tshirt_img.shape[1]))

        # Resize T-shirt
        tshirt_resized = cv2.resize(tshirt_img, (tshirt_width, tshirt_height))

        # Calculate position to overlay
        x1 = left_shoulder[0] - tshirt_width // 4
        y1 = left_shoulder[1]
        x2 = x1 + tshirt_width
        y2 = y1 + tshirt_height

        # Overlay T-shirt
        if 0 <= x1 < w and 0 <= y1 < h and x2 < w and y2 < h:
            alpha_tshirt = tshirt_resized[:, :, 3] / 255.0
            alpha_frame = 1.0 - alpha_tshirt

            for c in range(3):
                frame[y1:y2, x1:x2, c] = (alpha_tshirt * tshirt_resized[:, :, c] +
                                          alpha_frame * frame[y1:y2, x1:x2, c])

    # Show frame
    cv2.imshow("Virtual Try-On", frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


