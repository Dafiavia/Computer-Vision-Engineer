import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import warnings

# Suppress warning agar log bersih
warnings.filterwarnings("ignore", category=UserWarning)

class FaceEngine:
    def __init__(self):
        print("[FaceEngine] Initializing...")
        self.device = torch.device('cpu')
        
        # --- LOGIKA SAFETY SWITCH ---
        self.can_quantize = False
        self.backend = None
        
        # Cek backend apa yang didukung oleh PyTorch di komputer ini
        supported_engines = torch.backends.quantized.supported_engines
        print(f"[FaceEngine] Supported Quantization Engines: {supported_engines}")

        # Prioritas Backend
        if 'qnnpack' in supported_engines:
            self.backend = 'qnnpack'
            # Kita set engine, tapi tetap siap fallback jika crash
            torch.backends.quantized.engine = 'qnnpack'
            self.can_quantize = True
        elif 'fbgemm' in supported_engines:
            self.backend = 'fbgemm'
            torch.backends.quantized.engine = 'fbgemm'
            self.can_quantize = True
        else:
            print("[FaceEngine] ⚠️ No quantization backend supported. Using Float32.")
            self.can_quantize = False
            
        # 1. MTCNN (Face Detector)
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device, keep_all=False
        )

        # 2. InceptionResnetV1 (Feature Extractor)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
    def quantize_model(self, calibration_images: list):
        """
        Mencoba melakukan Static Quantization.
        Jika gagal (karena Windows/Arsitektur Model), otomatis revert ke Float32.
        """
        if not self.can_quantize:
            print("[FaceEngine] Skipping quantization. Running in Float32 mode.")
            return

        if not calibration_images:
            return

        print(f"[FaceEngine] Attempting quantization with {len(calibration_images)} samples...")
        
        try:
            # Simpan state asli model (untuk restore jika gagal)
            original_state = self.model.state_dict()
            
            self.model.eval()
            
            # [FIX] Hapus freeze_bn_stats yang bermasalah pada InceptionResnet
            # self.model.apply(torch.nn.intrinsic.modules.fused.bn_relu.freeze_bn_stats) <-- DIHAPUS
            
            self.model.qconfig = torch.quantization.get_default_qconfig(self.backend)
            torch.quantization.prepare(self.model, inplace=True)

            # Calibration Loop
            with torch.no_grad():
                for img in calibration_images:
                    try:
                        if img.mode != 'RGB': img = img.convert('RGB')
                        tensor = self.mtcnn(img)
                        if tensor is not None:
                            self.model(tensor.unsqueeze(0).to(self.device))
                    except:
                        pass

            torch.quantization.convert(self.model, inplace=True)
            print("[FaceEngine] Quantization Complete (Int8 Mode).")
            
        except Exception as e:
            print(f"[FaceEngine] ⚠️ Quantization Failed/Unstable: {e}")
            print("[FaceEngine] Reverting to Standard Float32 Model...")
            
            # Re-init model baru yang bersih (Float32)
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def extract_embedding(self, image: Image.Image) -> dict:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        boxes, _ = self.mtcnn.detect(image)
    
    # Ambil box pertama yang terdeteksi, jika ada (konversi ke list Python)
        face_bbox = boxes[0].tolist() if boxes is not None and len(boxes) > 0 else None
    # ----------------------------------------

        try:
            img_cropped, prob = self.mtcnn(image, return_prob=True)
        except:
             return { "found": False, "embedding": None, "confidence": 0, "message": "Detection Error"}

        if img_cropped is None:
            return {
                "found": False, 
                "embedding": None, 
                "confidence": 0.0,
                "message": "Wajah tidak terdeteksi"
            }

        img_tensor = img_cropped.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor)
        
        return {
            "found": True,
            "embedding": embedding.cpu().numpy().flatten(),
            "confidence": prob,
            "face_bbox": face_bbox,
            "message": "Success"
        }