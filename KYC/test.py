import pandas as pd
import numpy as np
import os
import sys

# --- SIMULASI KELAS DATABASE ---
class Database:
    def __init__(self):
        # PASTIKAN PATH INI SAMA DENGAN YANG ADA DI src/database.py
        self.EMBEDDING_FILE = "data/embeddings.csv" 
        
    def load_embeddings(self):
        if not os.path.exists(self.EMBEDDING_FILE) or os.path.getsize(self.EMBEDDING_FILE) == 0:
            print("[DB Load Test] WARNING: CSV file is empty or missing.")
            return [], [], np.array([]) 

        try:
            print("[DB Load Test] Reading CSV file...")
            # Membaca data tanpa header
            df = pd.read_csv(self.EMBEDDING_FILE, header=None)
            
            # Memastikan DataFrame tidak kosong
            if df.empty:
                print("[DB Load Test] WARNING: DataFrame is empty after reading.")
                return [], [], np.array([])

            # 1. Ekstrak ID dan Nama (Kolom 0 dan 1)
            ids = df.iloc[:, 0].astype(str).tolist()
            names = df.iloc[:, 1].astype(str).tolist()
            
            # 2. Ekstrak Vektor (Kolom 2 hingga akhir)
            # Menggunakan .values.astype(np.float32) untuk konversi paksa ke float
            db_vectors = df.iloc[:, 2:].values.astype(np.float32)
            
            print(f"✅ SUCCESS: Loaded {len(ids)} embeddings.")
            print(f"Sample Vector Shape: {db_vectors.shape}")
            
            return ids, names, db_vectors

        except Exception as e:
            # Ini akan mencetak traceback yang tersembunyi
            print("\n===========================================================")
            print("❌ FATAL ERROR: Database Load Failed.")
            print(f"Error Detail: {e}")
            print("===========================================================")
            import traceback
            traceback.print_exc()
            return [], [], np.array([])

# --- JALANKAN TES ---
print("--- STARTING DATABASE ISOLATION TEST ---")
db = Database()
db.load_embeddings()