import cv2
import os

# --- Configuration ---
image_folder = 'raw_cotton_images'
label_folder = 'labels'
annotated_folder = 'annotated_images'
os.makedirs(label_folder, exist_ok=True)
os.makedirs(annotated_folder, exist_ok=True)

images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
images.sort()

cv2.namedWindow('Annotator', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Annotator', 900, 900)

# Generate distinct colors for up to 50 IDs
def get_color(boll_id):
    colors = [
        (0,255,0),(0,0,255),(255,0,0),(0,255,255),(255,0,255),(255,255,0),
        (128,255,0),(0,128,255),(255,0,128),(128,0,255),(0,255,128),(255,128,0),
        (64,255,64),(64,64,255),(255,64,64),(0,200,200),(200,0,200),(200,200,0),
        (100,255,200),(200,100,255),(255,200,100),(0,150,255),(255,150,0),(150,0,255),
        (50,255,50),(50,50,255),(255,50,50),(0,180,100),(100,0,180),(180,100,0),
    ]
    return colors[boll_id % len(colors)]

def draw_labels(img, labels):
    temp = img.copy()
    h, w = img.shape[:2]
    for (boll_id, cx, cy, bw, bh) in labels:
        ix = int((cx - bw/2) * w)
        iy = int((cy - bh/2) * h)
        iw = int(bw * w)
        ih = int(bh * h)
        color = get_color(boll_id)
        cv2.rectangle(temp, (ix, iy), (ix+iw, iy+ih), color, 3)
        # Label background for readability
        label_text = f"Boll {boll_id}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(temp, (ix, iy-th-12), (ix+tw+6, iy), color, -1)
        cv2.putText(temp, label_text, (ix+3, iy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
    return temp

def draw_ui(img, labels, status_msg):
    temp = draw_labels(img, labels)
    lines = [
        f"Boxes: {len(labels)}   |   SPACE: draw box   |   U: undo   |   S: save & next   |   Q: quit",
        status_msg
    ]
    for i, line in enumerate(lines):
        cv2.putText(temp, line, (10, 30 + i*38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,0), 2)
    return temp

img_index = 0
status_msg = "Press SPACE to draw a box"

while img_index < len(images):
    img_name = images[img_index]
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    current_labels = []
    status_msg = "Press SPACE to draw a box"

    print(f"\n{'='*50}")
    print(f"Image {img_index+1}/{len(images)}: {img_name}")
    print("SPACE=draw box | U=undo | S=save & next | Q=quit")
    print('='*50)

    while True:
        view = draw_ui(img, current_labels, status_msg)
        cv2.imshow('Annotator', view)
        key = cv2.waitKey(30) & 0xFF

        # --- SPACE: draw ROI then ask for boll ID in terminal ---
        if key == ord(' '):
            status_msg = "Drawing box... (drag to select, ENTER to confirm, C to cancel)"
            view = draw_ui(img, current_labels, status_msg)
            roi = cv2.selectROI('Annotator', view, fromCenter=False, printNotice=False)
            rx, ry, rw, rh = roi

            if rw > 0 and rh > 0:
                # Show the drawn box while waiting for input
                preview = draw_labels(img, current_labels)
                ix, iy = rx, ry
                cv2.rectangle(preview, (ix, iy), (ix+rw, iy+rh), (255, 255, 255), 3)
                cv2.putText(preview, "Type boll ID in terminal...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,0), 2)
                cv2.imshow('Annotator', preview)
                cv2.waitKey(1)  # Force window refresh

                # Get boll ID from terminal
                while True:
                    try:
                        raw = input(f"  Enter boll ID for this box (or 'skip'): ").strip().lower()
                        if raw == 'skip':
                            status_msg = "Box skipped"
                            break
                        boll_id = int(raw)
                        cx = (rx + rw/2) / w
                        cy = (ry + rh/2) / h
                        bw_norm = rw / w
                        bh_norm = rh / h
                        current_labels.append((boll_id, cx, cy, bw_norm, bh_norm))
                        status_msg = f"Added Boll {boll_id} — {len(current_labels)} boxes total"
                        print(f"  ✓ Boll {boll_id} saved")
                        break
                    except ValueError:
                        print("  ✗ Invalid — enter a number or 'skip'")
            else:
                status_msg = "No box drawn — try again"

        # --- U: undo last box ---
        elif key == ord('u'):
            if current_labels:
                removed = current_labels.pop()
                status_msg = f"Removed Boll {removed[0]} — {len(current_labels)} boxes left"
                print(f"  Undo: removed Boll {removed[0]}")
            else:
                status_msg = "Nothing to undo"

        # --- S: save txt + annotated jpg, move to next ---
        elif key == ord('s'):
            # Save YOLO txt
            txt_name = img_name.replace('.jpg', '.txt')
            txt_path = os.path.join(label_folder, txt_name)
            with open(txt_path, 'w') as f:
                for (boll_id, cx, cy, bw, bh) in current_labels:
                    f.write(f"{boll_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            # Save annotated JPG (full resolution with boxes burned in)
            annotated = draw_labels(img, current_labels)
            ann_path = os.path.join(annotated_folder, img_name)
            cv2.imwrite(ann_path, annotated)

            print(f"  ✓ Labels saved  → {txt_path}")
            print(f"  ✓ Annotated img → {ann_path}")
            status_msg = "Saved! Loading next image..."
            img_index += 1
            break

        # --- Q: quit ---
        elif key == ord('q'):
            print("\nQuitting.")
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("\nAll images processed.")