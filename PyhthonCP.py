import cv2
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
      
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

# MediaPipe task initialization
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pose_landmarker.task')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f'Model file not found: {MODEL_PATH}. Please place pose_landmarker.task next to this script.')


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def download_youtube_video(url, status_label, download_button):
    try:
        status_label.config(text='Downloading from YouTube...', fg='orange')
        download_button.config(state=tk.DISABLED)
        
        output_path = os.path.join(os.path.dirname(__file__), 'youtube_video.mp4')
        
        cmd = [
            'C:/Program Files/Python314/python.exe', '-m', 'yt_dlp',
            '-f', 'best',
            '-o', output_path,
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(output_path):
            status_label.config(text=f'Downloaded: {output_path}', fg='green')
            download_button.config(state=tk.NORMAL)
            return output_path
        else:
            messagebox.showerror('Download Error', f'Failed to download video:\n{result.stderr}')
            status_label.config(text='Download failed.', fg='red')
            download_button.config(state=tk.NORMAL)
            return None
    except Exception as e:
        messagebox.showerror('Error', f'Download error:\n{str(e)}')
        status_label.config(text='Download error occurred.', fg='red')
        download_button.config(state=tk.NORMAL)
        return None


def run_pose_analysis(video_path, status_label, run_button):
    if not os.path.exists(video_path):
        messagebox.showerror('Error', f'Video file not found:\n{video_path}')
        run_button.config(state=tk.NORMAL)
        return

    # Create landmarker for this video analysis
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror('Error', f'Unable to open video:\n{video_path}')
        run_button.config(state=tk.NORMAL)
        return

    status_label.config(text='Processing video...')

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        feedback = 'Good Posture'

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            h, w, _ = frame.shape

            left_shoulder = [landmarks[11].x, landmarks[11].y]
            left_elbow = [landmarks[13].x, landmarks[13].y]
            left_wrist = [landmarks[15].x, landmarks[15].y]
            left_hip = [landmarks[23].x, landmarks[23].y]
            left_knee = [landmarks[25].x, landmarks[25].y]
            left_ankle = [landmarks[27].x, landmarks[27].y]

            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            if knee_angle > 170:
                feedback = 'Bend knees more'
            elif elbow_angle < 40:
                feedback = 'Improve racket angle'
            elif elbow_angle > 160:
                feedback = 'Weak follow-through'
            else:
                feedback = 'Excellent Movement'

            drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
            )

            cv2.putText(frame, f'Elbow Angle: {int(elbow_angle)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f'Knee Angle: {int(knee_angle)}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.rectangle(frame, (10, frame.shape[0] - 80), (520, frame.shape[0] - 20), (0, 0, 0), -1)
        cv2.putText(frame, f'Feedback: {feedback}', (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Pose Estimation - Coaching AI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    status_label.config(text='Analysis completed.')
    run_button.config(state=tk.NORMAL)


class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Pose Analysis UI')
        self.video_path = ''

        self.select_button = tk.Button(root, text='Select Video', command=self.select_video, width=20)
        self.select_button.pack(pady=10)

        self.path_label = tk.Label(root, text='No video selected', wraplength=400)
        self.path_label.pack(padx=10, pady=5)

        # YouTube section
        tk.Label(root, text='Or Download from YouTube:', font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        self.youtube_entry = tk.Entry(root, width=60)
        self.youtube_entry.pack(padx=10, pady=5)
        self.youtube_entry.insert(0, 'Paste YouTube URL here...')
        
        self.download_button = tk.Button(root, text='Download from YouTube', command=self.download_youtube, width=20)
        self.download_button.pack(pady=5)

        self.run_button = tk.Button(root, text='Run Analysis', command=self.start_analysis, width=20, state=tk.DISABLED)
        self.run_button.pack(pady=10)

        self.status_label = tk.Label(root, text='Waiting for video selection...', fg='blue')
        self.status_label.pack(pady=5)

    def select_video(self):
        selected_path = filedialog.askopenfilename(
            title='Select video file',
            filetypes=[('Video files', '*.mp4 *.avi *.mov *.mkv'), ('All files', '*.*')]
        )
        if selected_path:
            self.video_path = selected_path
            self.path_label.config(text=self.video_path)
            self.run_button.config(state=tk.NORMAL)
            self.status_label.config(text='Video selected. Ready to run.')

    def download_youtube(self):
        url = self.youtube_entry.get().strip()
        if not url or url == 'Paste YouTube URL here...':
            messagebox.showwarning('No URL', 'Please paste a YouTube URL first.')
            return
        
        threading.Thread(target=self._download_thread, args=(url,), daemon=True).start()

    def _download_thread(self, url):
        video_path = download_youtube_video(url, self.status_label, self.download_button)
        if video_path:
            self.video_path = video_path
            self.path_label.config(text=f'Downloaded: {video_path}')
            self.run_button.config(state=tk.NORMAL)
            self.status_label.config(text='Video ready. Click Run Analysis to start.', fg='green')

    def start_analysis(self):
        if not self.video_path:
            messagebox.showwarning('No Video', 'Please select a video file first.')
            return

        self.run_button.config(state=tk.DISABLED)
        self.status_label.config(text='Starting analysis...', fg='blue')
        threading.Thread(target=run_pose_analysis, args=(self.video_path, self.status_label, self.run_button), daemon=True).start()


if __name__ == '__main__':
    root = tk.Tk()
    app = PoseApp(root)
    root.geometry('650x380')
    root.mainloop()
 
