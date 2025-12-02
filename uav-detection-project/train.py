from ultralytics import YOLO

def train_model():
    # 1. Load model DASAR (yolo11n.pt), BUKAN model lama (best.pt)
    model = YOLO("yolo11n.pt") 

    # 2. Train dengan nama project baru agar mudah dibedakan
    results = model.train(
        data="datasets/uav_dataset/data.yaml", 
        epochs=150,             # Naikkan epoch karena data banyak
        patience=30,            # Early stopping
        imgsz=640,
        batch=16,               # Sesuaikan VRAM
        device=0,               
        project="results",      
        name="uav_yolo11n_v2", # Beri nama BEDA biar tidak bingung
        exist_ok=False          # Pastikan membuat folder baru
    )

if __name__ == "__main__":
    train_model() 