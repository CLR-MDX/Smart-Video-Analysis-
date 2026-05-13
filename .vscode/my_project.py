# ============================================================
# FormOle - Video Analysis Coaching AI
# Beginner Friendly Version
# ============================================================
# FEATURES:
#  Upload gameplay video
#  Detect body pose using MediaPipe
#  Draw skeleton tracking
#  Analyze posture
#  Give coaching feedback
#  Save analyzed output video
#
# REQUIRED LIBRARIES:
# pip install opencv-python mediapipe numpy
#
# RUN:
# python formole.py
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
import math

# =========================
# MEDIAPIPE SETUP
# =========================
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='../pose_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = PoseLandmarker.create_from_options(options)

# =========================
# ANGLE CALCULATION
# =========================
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# =========================
# VIDEO INPUT
# =========================
video_path = "../video.mp4"

cap = cv2.VideoCapture(video_path)

# =========================
# OUTPUT VIDEO
# =========================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_analysis.avi', fourcc, fps, (width, height))

print("\nProcessing Video...\n")

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # Flip for better view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect Pose
    results = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    feedback = "Good Posture"

    # =========================
    # LANDMARK ANALYSIS
    # =========================
    if results.pose_landmarks:

        landmarks = results.pose_landmarks[0]

        # Get Coordinates
        left_shoulder = [
            landmarks[11].x,
            landmarks[11].y
        ]

        left_elbow = [
            landmarks[13].x,
            landmarks[13].y
        ]

        left_wrist = [
            landmarks[15].x,
            landmarks[15].y
        ]

        left_hip = [
            landmarks[23].x,
            landmarks[23].y
        ]

        left_knee = [
            landmarks[25].x,
            landmarks[25].y
        ]

        left_ankle = [
            landmarks[27].x,
            landmarks[27].y
        ]

        # =========================
        # CALCULATE ANGLES
        # =========================
        elbow_angle = calculate_angle(
            left_shoulder,
            left_elbow,
            left_wrist
        )

        knee_angle = calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        # =========================
        # FEEDBACK SYSTEM
        # =========================
        if knee_angle > 170:
            feedback = "Bend knees more"

        elif elbow_angle < 40:
            feedback = "Improve racket angle"

        elif elbow_angle > 160:
            feedback = "Weak follow-through"

        else:
            feedback = "Excellent Movement"

        # =========================
        # DISPLAY ANGLES
        # =========================
        cv2.putText(
            frame,
            f'Elbow Angle: {int(elbow_angle)}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f'Knee Angle: {int(knee_angle)}',
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        # =========================
        # DRAW SKELETON
        # =========================
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=results.pose_landmarks[0],
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=(0, 255, 255),
                thickness=2,
                circle_radius=2
            )
        )

    # =========================
    # FEEDBACK DISPLAY
    # =========================
    cv2.rectangle(frame, (10, height - 80), (500, height - 20), (0,0,0), -1)

    cv2.putText(
        frame,
        f'Feedback: {feedback}',
        (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # =========================
    # SHOW VIDEO
    # =========================
    cv2.imshow("FormOle - Coaching AI", frame)

    # Save Output
    out.write(frame)
 
    # Exit Button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE EVERYTHING
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()
landmarker.close()

print("\n================================")
print("Analysis Completed Successfully")
print("Output Saved as output_analysis.avi")
print("================================")