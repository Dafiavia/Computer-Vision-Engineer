import easyocr
from PIL import Image
import numpy as np
from typing import List, Dict

class OCREngine:
    def __init__(self):
        print("[OCREngine] Initializing EasyOCR Reader (Lang: EN, ES)...")
        self.is_ready = False 
        
        try:
            # Menggunakan model multilingual (English & Spanish)
            self.reader = easyocr.Reader(['en', 'es'], gpu=False)
            self.is_ready = True
            print("[OCREngine] EasyOCR Model Loaded Successfully.")
        except Exception as e:
            print(f"[OCREngine] FATAL ERROR LOADING READER: {e}")

    def extract_text(self, image: Image.Image) -> List[Dict]:
        """
        Melakukan OCR dan mengembalikan list terstruktur [box, text, confidence].
        """
        if not self.is_ready:
            raise RuntimeError("OCR Model is not initialized.")

        # Konversi PIL Image ke Numpy Array
        image_np = np.array(image.convert('RGB'))
        
        try:
            # EasyOCR mengembalikan: [[bbox], teks, confidence]
            results = self.reader.readtext(image_np, detail=1) 
            
            structured_output = []
            
            for (bbox, text, conf) in results:
                # [FIX]: Pastikan confidence dikonversi ke float standar Python
                structured_output.append({
                    "box": bbox, # Koordinat Bounding Box (Masih perlu konversi di Parser)
                    "text": text,
                    "conf": float(conf) # Konversi ke float standar
                })
            
            return structured_output
            
        except Exception as e:
            raise RuntimeError(f"EasyOCR runtime error: {str(e)}")