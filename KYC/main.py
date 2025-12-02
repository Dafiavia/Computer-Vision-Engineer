import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- PENTING: Import CORS
from PIL import Image
import io
import logging

# Import pipeline logic dari folder src
from src.pipeline import KYCPipeline 

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inisialisasi Aplikasi
app = FastAPI(
    title="KYC Engine (Spanish DNI)",
    description="Engine for OCR Parsing and Face Verification.",
    version="0.1.0"
)

# --- KONFIGURASI CORS (SANGAT PENTING) ---
# Ini mengizinkan frontend (HTML) untuk berbicara dengan backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua origin (aman untuk development lokal)
    allow_credentials=True,
    allow_methods=["*"],  # Mengizinkan semua method (POST, GET, OPTIONS)
    allow_headers=["*"],  # Mengizinkan semua headers
)

# Inisialisasi Pipeline KYC
try:
    pipeline = KYCPipeline()
    logger.info("✅ Pipeline initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize pipeline: {e}")
    raise e

# --- ENDPOINTS ---

@app.get("/")
def root():
    return {"message": "KYC Engine is Running & CORS is Enabled!"}

# 1. Registrasi ID Card
@app.post("/register-idcard")
async def register_idcard(image: UploadFile = File(...)):
    # Validasi tipe file
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
        
    try:
        # Baca file gambar
        contents = await image.read()
        image_pil = Image.open(io.BytesIO(contents))
        
        # Proses dengan Pipeline
        logger.info("Processing ID Card...")
        result = pipeline.process_id_card(image_pil)
        return result
        
    except Exception as e:
        logger.error(f"Error processing ID card: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# 2. Registrasi Wajah
@app.post("/register-face")
async def register_face(id_number: str = Form(...), image: UploadFile = File(...)):
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
        
    try:
        contents = await image.read()
        image_pil = Image.open(io.BytesIO(contents))
        
        # Bersihkan ID Number
        clean_id = id_number.strip().upper()
        
        # Ambil Nama Lengkap dari Database (karena frontend tidak mengirim nama)
        user_data = pipeline.db.get_user_by_id(clean_id)
        if not user_data:
            return {
                "status": "error", 
                "message": f"ID {clean_id} tidak ditemukan. Harap registrasi ID Card terlebih dahulu."
            }
        
        full_name = user_data.get("full_name", "Unknown")
        
        # Proses Registrasi Wajah
        logger.info(f"Registering face for {clean_id} ({full_name})...")
        result = pipeline.process_face_registration(image_pil, clean_id, full_name)
        return result

    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# 3. Verifikasi KYC
@app.post("/kyc-check")
async def kyc_check(image: UploadFile = File(...)):
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
        
    try:
        contents = await image.read()
        image_pil = Image.open(io.BytesIO(contents))
        
        # Proses Matching
        logger.info("Performing KYC Check...")
        result = pipeline.kyc_match(image_pil)
        return result
        
    except Exception as e:
        logger.error(f"Error KYC check: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Entry point untuk debugging langsung
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)