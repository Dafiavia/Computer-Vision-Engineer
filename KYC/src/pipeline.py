import os
import glob
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# Import modul lokal
from .face_engine import FaceEngine
from .ocr_engine import OCREngine
from .database import Database
from .id_parser import IDParser 

class KYCPipeline:
    def __init__(self):
        print("--- Initializing KYC Pipeline ---")
        self.db = Database()
        self.ocr_engine = OCREngine()
        self.id_parser = IDParser() 
        self.face_engine = FaceEngine()
        
        self._run_initial_calibration()

    def _run_initial_calibration(self):
        calibration_imgs = []
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
        calib_files = []
        
        for ext in extensions:
            pattern = f"dataset/face/**/{ext}"
            calib_files.extend(glob.glob(pattern, recursive=True))
        
        calib_files = calib_files[:50]

        if calib_files:
            print(f"[Pipeline] Found {len(calib_files)} images in dataset for calibration.")
            for f in calib_files:
                try:
                    img = Image.open(f).convert('RGB')
                    calibration_imgs.append(img)
                except:
                    pass
        
        if len(calibration_imgs) < 5:
            print("[Pipeline] Dataset not found or empty. Using Random Noise for calibration.")
            for _ in range(10):
                arr = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
                calibration_imgs.append(Image.fromarray(arr))
                
        self.face_engine.quantize_model(calibration_imgs)
        del calibration_imgs 

    def process_id_card(self, image: Image.Image):
        """Flow 1: Upload DNI -> OCR -> PARSE -> Save JSON"""
        
        try:
            raw_data = self.ocr_engine.extract_text(image)
        except RuntimeError as e:
            return {"status": "error", "message": f"OCR Runtime Failed: {str(e)}"}
            
        data = self.id_parser.parse_data(raw_data) 
        
        if not data.get('id_number'):
            return {"status": "failed", "message": "Gagal membaca Nomer ID (DNI) atau format tidak ditemukan."}
            
        success, msg = self.db.save_ocr_result(data)
        
        status = "success" if success else "failed"
        if success and not data.get('id_valid', False):
             msg += " (Warning: Checksum ID tidak valid, kemungkinan DNI palsu/salah baca)"

        return {"status": status, "message": msg, "data": data}

    def process_face_registration(self, image: Image.Image, id_number: str, full_name: str):
        """Flow 2: Upload Wajah -> Detect -> Save CSV (Linked by ID)"""
        
        user_ocr = self.db.get_user_by_id(id_number)
        if not user_ocr:
            return {"status": "failed", "message": f"ID {id_number} belum terdaftar. Silakan upload DNI dulu."}

        result = self.face_engine.extract_embedding(image)
        
        if not result['found']:
            return {"status": "failed", "message": "Wajah tidak terdeteksi. Gunakan foto selfie yang jelas."}
        
        self.db.save_embedding(id_number, full_name, result['embedding'])
        return {"status": "success", "message": "Wajah berhasil didaftarkan."}

    def kyc_match(self, image: Image.Image):
        """Flow 3: Login/Check -> Match Face -> Return User Data + Face Bounding Box"""
        
        result = self.face_engine.extract_embedding(image)
        if not result['found']:
            # Tambahkan respons bounding box kosong jika wajah tidak ditemukan
            return {"status": "error", "message": "Wajah tidak terdeteksi", "face_bbox": None} 

        query_vec = result['embedding'].reshape(1, -1)
        
        ids, names, db_vectors = self.db.load_embeddings()
        if len(ids) == 0:
            # Tambahkan respons bounding box kosong jika database kosong
            return {"status": "error", "message": "Database wajah kosong", "face_bbox": None}

        sim_scores = cosine_similarity(query_vec, db_vectors)[0]
        best_idx = np.argmax(sim_scores)
        score = float(sim_scores[best_idx])
        
        THRESHOLD = 0.65
        is_match = score > THRESHOLD
        
        # --- PERUBAHAN: Tambahkan face_bbox ke response ---
        resp = {
            "status": "success",
            "match": is_match,
            "similarity_score": f"{score:.4f}",
            "candidate_name": names[best_idx] if is_match else None,
            "user_details": None,
            "face_bbox": result['face_bbox'] if result['face_bbox'] is not None else None # Pastikan dikonversi ke list
        }
        # ----------------------------------------------------

        if is_match:
            matched_id = str(ids[best_idx]).strip().upper() 
            user_data = self.db.get_user_by_id(matched_id)
            resp['user_details'] = user_data
            
        return resp