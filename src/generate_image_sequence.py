import cv2
import os

def video_to_images(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        name = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(name, frame)
        
        frame_count += 1

    cap.release()
    print(f"Done! {frame_count} frames saved in '{output_folder}'.")
