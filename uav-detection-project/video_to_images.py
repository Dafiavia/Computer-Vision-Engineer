import cv2
import os
import glob
import re # Diperlukan untuk mengekstrak angka dari nama file

def find_last_index(output_folder):
    """Mencari nomor indeks tertinggi dari file yang sudah ada."""
    
    # Pola file yang kita cari: uav_data_XXX.jpg
    search_pattern = os.path.join(output_folder, "uav_data_*.jpg")
    
    # Gunakan glob untuk menemukan semua file yang cocok
    existing_files = glob.glob(search_pattern)
    
    if not existing_files:
        # Jika tidak ada file, mulai dari 0
        return 0
    
    max_index = 0
    
    # Gunakan Regular Expression (regex) untuk menemukan angka di nama file
    # Contoh: uav_data_123.jpg -> ekstrak 123
    # Catatan: Gunakan pola yang cocok dengan format penamaan Anda: uav_data_
    pattern = re.compile(r"uav_data_(\d+)\.jpg$")
    
    for filename in existing_files:
        match = pattern.search(filename)
        if match:
            # Mengambil angka yang ditemukan (grup 1)
            index = int(match.group(1))
            if index > max_index:
                max_index = index
                
    # Kita akan mulai menimpa pada indeks tertinggi + 1
    return max_index + 1

def extract_frames_every_second(video_path, output_folder):
    
    # --- Modifikasi Bagian 1: Inisialisasi Penomeran ---
    
    # 1. Cek apakah file video ada
    if not os.path.exists(video_path):
        print(f"Error: File video '{video_path}' tidak ditemukan.")
        return

    # 2. Buka Video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Tidak bisa membuka video.")
        return

    # 3. Dapatkan FPS (Frame Per Second)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video FPS: {fps}")

    # 4. Buat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 5. Cari indeks terakhir untuk melanjutkan penomeran! 🚀
    current_index = find_last_index(output_folder)
    print(f"Ditemukan {current_index} file yang sudah ada. Penomeran baru dimulai dari indeks {current_index}.")

    frame_count = 0
    saved_count = 0

    print("Mulai mengambil screenshot...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Logika: Ambil frame setiap kelipatan FPS (Setiap 1 Detik)
        if frame_count % fps == 0:
            
            # --- Modifikasi Bagian 2: Penomeran File ---
            # second = int(frame_count / fps) # Kita tidak lagi menggunakan detik video
            
            # Gunakan penomeran yang berkelanjutan (current_index)
            filename = f"{output_folder}/uav_data_{current_index}.jpg" 
            
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"Disimpan: {filename}")
            
            # Setelah disimpan, tingkatkan current_index untuk file berikutnya
            current_index += 1 

        frame_count += 1

    # Bersihkan memori
    cap.release()
    print(f"\nSelesai! Total {saved_count} gambar BARU tersimpan di folder '{output_folder}'.")

# --- KONFIGURASI ---
if __name__ == "__main__":
    # GANTI 'video_saya.mp4' dengan path video drone Anda
    VIDEO_INPUT = "datasets/night_chase_uav.mp4" 
    
    # Folder tempat gambar akan disimpan
    FOLDER_OUTPUT = "datasets/new_extracted_images"

    extract_frames_every_second(VIDEO_INPUT, FOLDER_OUTPUT)