from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import requests
import os
import urllib.parse

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

@app.route('/process-video', methods=['POST'])
def process_video():
    data = request.get_json()
    video_url = data.get('video_url')
    
    if not video_url:
        return jsonify({'error': 'No video URL provided'}), 400

    try:
        # Download video from Cloudinary
        response = requests.get(video_url, stream=True)
        input_path = 'temp_input.mp4'
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        # Process video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video'}), 500

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Output video setup
        output_path = 'temp_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Draw stickman (pose landmarks)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Simple feedback based on pose (example: check elbow angle for bicep curl)
                landmarks = results.pose_landmarks.landmark
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

                # Calculate angle (simplified)
                angle = abs(left_elbow.y - left_shoulder.y) * 180
                feedback = "Good form!" if angle < 0.5 else "Keep elbows higher!"

                # Add feedback text
                cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        # Upload processed video to Cloudinary using unsigned preset
        with open(output_path, 'rb') as video_file:
            files = {'file': (os.path.basename(output_path), video_file, 'video/mp4')}
            data = {
                'upload_preset': 'processed_workout_upload',  # Your unsigned upload preset for server
                'folder': 'processed_videos'
            }
            response = requests.post(
                'https://api.cloudinary.com/v1_1/YOUR_CLOUD_NAME/video/upload',  # Replace YOUR_CLOUD_NAME
                files=files,
                data=data
            )

        if response.status_code != 200:
            return jsonify({'error': 'Failed to upload processed video to Cloudinary'}), 500

        upload_result = response.json()
        feedback_video_url = upload_result['secure_url']

        # Clean up
        os.remove(input_path)
        os.remove(output_path)

        return jsonify({'feedback_video_url': feedback_video_url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))