import pandas as pd
import numpy as np
import os
import json # <--- IMPORT TAMBAHAN
from typing import List

class Database:
    def __init__(self):
        self.EMBEDDING_FILE = "data/embeddings.csv" 
        self.OCR_FILE = "data/id_data.json"
        
    # --- METHOD UNTUK MENGAMBIL DATA OCR SAAT KYCMATCH ---
    def get_user_by_id(self, id_number: str) -> dict | None:
        """
        Mengambil data OCR pengguna dari JSON berdasarkan ID Number.
        """
        # os.path.exists membutuhkan import os
        if not os.path.exists(self.OCR_FILE):
            return None
        
        try:
            with open(self.OCR_FILE, 'r') as f:
                data = json.load(f)
                
            # ID tersimpan sebagai key utama di JSON
            return data.get(id_number)
            
        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    # --- METHOD UNTUK MENYIMPAN EMBEDDING ---
    def save_embedding(self, id_number, full_name, embedding_vector):
        """Menyimpan embedding baru ke CSV tanpa header dan tanpa index."""
        
        vector_list = embedding_vector.tolist() 
        row = [id_number, full_name] + vector_list
        
        df_new = pd.DataFrame([row])
        
        df_new.to_csv(
            self.EMBEDDING_FILE, 
            mode='a', 
            header=False, 
            index=False,
        )

    # --- METHOD UNTUK MEMUAT SEMUA EMBEDDING ---
    def load_embeddings(self):
        """Memuat data embeddings dari embeddings.csv."""
        
        if not os.path.exists(self.EMBEDDING_FILE) or os.path.getsize(self.EMBEDDING_FILE) == 0:
            return [], [], np.array([]) 

        try:
            df = pd.read_csv(self.EMBEDDING_FILE, header=None)
            
            if df.empty:
                return [], [], np.array([])

            # Ekstrak ID dan Nama (Kolom 0 dan 1)
            ids = df.iloc[:, 0].astype(str).tolist()
            names = df.iloc[:, 1].astype(str).tolist()
            
            # Ekstrak Vektor (Kolom 2 hingga akhir)
            db_vectors = df.iloc[:, 2:].values.astype(np.float32)
            
            return ids, names, db_vectors

        except Exception as e:
            print(f"DATABASE CRITICAL FAILURE: Load Embeddings Failed. Error: {e}")
            return [], [], np.array([])

    # --- METHOD UNTUK MENYIMPAN HASIL OCR ---
    def save_ocr_result(self, data: dict):
        """Menyimpan hasil OCR ke id_data.json"""
        id_number_raw = data.get('id_number')
        
        # --- FIX FINAL: Bersihkan ID saat digunakan sebagai key ---
        clean_id = str(id_number_raw).strip().upper() 
        # --------------------------------------------------------

        # 1. Cek duplikasi (menggunakan clean_id)
        if self.get_user_by_id(clean_id):
            return False, f"ID {id_number_raw} sudah terdaftar."

        # 2. Muat data lama
        if os.path.exists(self.OCR_FILE):
            with open(self.OCR_FILE, 'r') as f:
                db_data = json.load(f)
                
            if not isinstance(db_data, dict):
                 db_data = {}
        else:
            db_data = {}
        
        # 3. Tambahkan data baru (Gunakan clean_id sebagai KEY dan pastikan value ID di dalamnya juga bersih)
        # Penting: Update data['id_number'] agar value-nya juga bersih
        data['id_number'] = clean_id
        db_data[clean_id] = data
        
        # 4. Simpan kembali
        try:
            with open(self.OCR_FILE, 'w') as f:
                json.dump(db_data, f, indent=4)
            return True, "Data ID Card berhasil disimpan."
        except Exception as e:
            return False, f"Gagal menyimpan data ke JSON: {str(e)}"