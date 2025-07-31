#!/usr/bin/env python3
import cv2
import os
import face_recognition
from PIL import Image
import numpy as np

def test_single_video(video_path, output_dir="test_faces", max_faces=5):
    """Test face extraction on a single video"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Testing face extraction on: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return False

    extracted_count = 0
    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.read()[0] and extracted_count < max_faces:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Sample every 30 frames
        if frame_count % 30 != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        except Exception as e:
            print(f"Error in face detection: {e}")
            continue

        for top, right, bottom, left in face_locations:
            if extracted_count >= max_faces:
                break

            # Add padding
            padding = 20
            h, w = rgb_frame.shape[:2]
            top = max(0, top - padding)
            right = min(w, right + padding)
            bottom = min(h, bottom + padding)
            left = max(0, left + padding)

            face_image = rgb_frame[top:bottom, left:right]
            if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                continue

            face_pil = Image.fromarray(face_image)
            face_resized = face_pil.resize((299, 299), Image.Resampling.LANCZOS)

            output_path = os.path.join(output_dir, f"{video_name}_face_{extracted_count}.jpg")
            face_resized.save(output_path, quality=95)
            extracted_count += 1
            print(f"  ✅ Extracted face {extracted_count}")

    cap.release()
    print(f"✅ Test complete: {extracted_count} faces extracted from {video_name}")
    return extracted_count > 0

print("�� Testing face extraction on sample videos...")
test_videos = [
    "Celeb-real/id0_0000.mp4",
    "Celeb-synthesis/id0_id16_0000.mp4",
    "YouTube-real/00000.mp4"
]

success_count = 0
for video_path in test_videos:
    if os.path.exists(video_path):
        if test_single_video(video_path):
            success_count += 1
    else:
        print(f"❌ Video not found: {video_path}")

if success_count > 0:
    print(f"\n🎉 Face extraction test successful! ({success_count}/{len(test_videos)} videos)")
    import glob
    sample_faces = glob.glob("test_faces/*.jpg")
    print("\nSample extracted faces:")
    for face in sample_faces[:5]:
        print(f"  - {face}")
else:
    print("❌ Face extraction test failed. Check dependencies.")
