from ultralytics import YOLO

# --- KONFIGURASI PATH ---
# GANTI dengan path ke model hasil training Anda
MODEL_PATH = "results/uav_yolo11n_v2/weights/best.pt" 

def evaluate_model():
    # 1. Load model yang sudah dilatih
    model = YOLO(MODEL_PATH)

    # 2. Lakukan Validasi pada data Test
    # 'split=test' memerintahkan Ultralytics menggunakan path yang ada di kunci 'test:' pada data.yaml
    results = model.val(
        data="datasets/uav_dataset/data.yaml", 
        split="valid", 
        imgsz=640,
        conf=0.25, # Ambang batas confidence standar
        iou=0.7,   # Ambang batas IOU standar
        device='cpu' # Gunakan 'cpu' atau '0' jika GPU tersedia
    )
    
    # 3. Cetak Hasil Utama
    print("\n--- HASIL EVALUASI RESMI PADA DATA TEST ---")
    print(f"mAP50 (High Accuracy Bounding Box): {results.box.map50}")
    print(f"mAP50-95 (Average Accuracy Bounding Box): {results.box.map}")
    print(f"Precision: {results.box.p}")
    print(f"Recall: {results.box.r}")

if __name__ == "__main__":
    evaluate_model()