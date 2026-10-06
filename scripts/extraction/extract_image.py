import cv2
import os

# --- Configuration ---
video_path = 'GX010049.MP4'
output_folder = 'raw_cotton_images'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_skip = 2  # Only display every Nth frame to reduce load

cv2.namedWindow('Extractor', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Extractor', 800, 800)

print("Controls: [s] Save Frame | [f] Fast-Forward 2s | [r] Rewind 2s | [q] Quit")

display_frame = None  # Keep last good frame for saving after freeze
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # --- Skip frames to reduce display load ---
    if frame_count % frame_skip != 0:
        continue

    # --- Rotate and Crop ---
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h, w = frame.shape[:2]
    start_y = (h - w) // 2
    cropped = frame[start_y:start_y + w, 0:w]
    display_frame = cropped  # Cache for saving

    # --- Resize for display only (not for saving) ---
    preview = cv2.resize(cropped, (800, 800), interpolation=cv2.INTER_AREA)
    cv2.imshow('Extractor', preview)

    # --- Increase waitKey delay so window events are processed ---
    key = cv2.waitKey(30) & 0xFF

    if key == ord('s'):
        if display_frame is not None:
            f_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            filename = os.path.join(output_folder, f"frame_{f_id:06d}.jpg")
            cv2.imwrite(filename, display_frame)  # Save full-res, not preview
            print(f"Saved: {filename}")

    elif key == ord('f'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + (fps * 2))

    elif key == ord('r'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - (fps * 2)))

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()