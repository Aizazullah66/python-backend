from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import cloudinary
import cloudinary.uploader
import urllib.request
import tempfile

app = Flask(__name__)

# Cloudinary Configuration (use your existing credentials, no api_secret)
cloudinary.config(
    cloud_name='dmwaesnu7',  # e.g., 'd123456789'
    api_key='929586615284361',        # e.g., '123456789012345'
    secure=True
)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def analyze_squat_form(keypoints):
    """Analyze squat form based on keypoints."""
    hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
    knee = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value]
    ankle = keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    vector1 = np.array([hip.x - knee.x, hip.y - knee.y])
    vector2 = np.array([ankle.x - knee.x, ankle.y - knee.y])
    angle = np.degrees(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))

    if angle < 90:
        return "Knees too far forward! Keep them over ankles.", (0, 0, 255)  # Red for incorrect
    return "Good form!", (0, 255, 0)  # Green for correct

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f'input_{timestamp}.mp4'  # File name without folder
    output_filename = f'output_{timestamp}.mp4'  # File name without folder

    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
        input_path = temp_input.name
        video_file.save(input_path)

    # Upload input video to Cloudinary (dynamic folder)
    try:
        upload_result = cloudinary.uploader.upload(
            input_path,
            resource_type="video",
            folder="workout_uploads",  # Dynamic folder
            public_id=input_filename,
            upload_preset='workoutvideos'  # e.g., 'workout_unsigned'
        )
        input_video_url = upload_result['secure_url']
    except Exception as e:
        os.remove(input_path)
        return jsonify({'error': f'Failed to upload to Cloudinary: {str(e)}'}), 500

    # Download video for processing
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_process:
        process_path = temp_process.name
        urllib.request.urlretrieve(input_video_url, process_path)

    # Process video
    cap = cv2.VideoCapture(process_path)
    if not cap.isOpened():
        os.remove(input_path)
        os.remove(process_path)
        return jsonify({'error': 'Failed to open video'}), 500

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
        output_path = temp_output.name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Draw stickman (skeleton with joints and connections)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Green joints
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green connections
                )

                # Analyze form and get feedback with color
                feedback, text_color = analyze_squat_form(results.pose_landmarks.landmark)

                # Add confidence score to confirm detection
                confidence = results.pose_landmarks.landmark[0].visibility  # Nose visibility
                confidence_text = f"Detection Confidence: {confidence:.2f}"

                # Overlay feedback text (centered at bottom)
                text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
                text_x = (frame_width - text_size[0]) // 2
                text_y = frame_height - 50  # Near bottom
                cv2.putText(frame, feedback, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2, cv2.LINE_AA)

                # Overlay confidence text (top-left)
                cv2.putText(frame, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)

        cap.release()
        out.release()

    # Upload processed video to Cloudinary (dynamic folder)
    try:
        upload_result = cloudinary.uploader.upload(
            output_path,
            resource_type="video",
            folder="processed/",  # Dynamic folder
            public_id=output_filename,
            upload_preset='YOUR_UNSIGNED_UPLOAD_PRESET'
        )
        video_url = upload_result['secure_url']
    except Exception as e:
        os.remove(input_path)
        os.remove(process_path)
        os.remove(output_path)
        return jsonify({'error': f'Failed to upload processed video to Cloudinary: {str(e)}'}), 500

    # Clean up temporary files
    os.remove(input_path)
    os.remove(process_path)
    os.remove(output_path)

    return jsonify({'videoUrl': video_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))