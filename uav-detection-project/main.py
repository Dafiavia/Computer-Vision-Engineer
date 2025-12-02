import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# --- KONFIGURASI ---
# Menggunakan model terbaik Anda (v2)
MODEL_PATH = "results/uav_yolo11n/weights/best.pt" 
VIDEO_SOURCE = "datasets/night_chase_uav.mp4" 
OUTPUT_PATH = "inference_output/result_clean_yolo_byte.mp4"

# --- HYPERPARAMETER ---
CONFIDENCE_THRESHOLD = 0.5  # Ambang batas kepercayaan (bisa diturunkan ke 0.3 jika kurang sensitif)
IOU_THRESHOLD = 0.5         # Ambang batas NMS

def main():
    # 1. Load Model
    try:
        print(f"Memuat model dari: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error memuat model: {e}")
        return

    # 2. Inisialisasi ByteTrack
    # Menggunakan setting default yang stabil
    print("Inisialisasi ByteTrack...")
    byte_tracker = sv.ByteTrack(frame_rate=30)

    # 3. Inisialisasi Visualisasi (BERSIH)
    # HANYA Kotak dan Label. TIDAK ADA Trace/Garis.
    box_annotator = sv.BoxAnnotator(
        thickness=2, 
        color_lookup=sv.ColorLookup.TRACK
    )
    
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5, 
        text_thickness=1, 
        color_lookup=sv.ColorLookup.TRACK,
        text_position=sv.Position.TOP_CENTER
    )

    # 4. Setup Video
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka video {VIDEO_SOURCE}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup Writer
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("Mulai Tracking Bersih (YOLO + ByteTrack). Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video selesai.")
            break

        # =======================================================
        # 1. DETEKSI (YOLO)
        # =======================================================
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
        
        # Konversi ke format Supervision
        detections = sv.Detections.from_ultralytics(results)

        # =======================================================
        # 2. TRACKING (BYTETRACK)
        # =======================================================
        # Update tracker dengan deteksi baru
        detections = byte_tracker.update_with_detections(detections)

        # =======================================================
        # 3. VISUALISASI
        # =======================================================
        annotated_frame = frame.copy()
        
        # Gambar Kotak
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        
        # Gambar Label (Isi: Tracker ID + Class Name + Confidence)
        labels = []
        for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence):
            class_name = model.names[class_id]
            labels.append(f"#{tracker_id} {class_name} {confidence:.2f}")

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections, 
            labels=labels
        )

        # Tampilkan
        cv2.imshow("Clean UAV Tracking", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()