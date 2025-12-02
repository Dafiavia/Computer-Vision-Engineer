import re
from typing import List, Dict

class IDParser:
    def __init__(self, confidence_threshold=0.60): 
        self.CONF_THRESHOLD = confidence_threshold
        
    def validate_dni_checksum(self, dni_full: str) -> bool:
        # Logika Checksum DNI Spanyol
        try:
            clean_dni = re.sub(r'[^0-9A-Z]', '', dni_full.upper())
            if len(clean_dni) != 9: return False
            numbers = int(clean_dni[:8])
            letter = clean_dni[-1]
            control = "TRWAGMYFPDXBNJZSQVHLCKE"
            return control[numbers % 23] == letter
        except:
            return False

    def parse_data(self, structured_ocr_output: List[Dict]) -> Dict:
        """
        Parsing Data DNI Spanyol (Adjacency Logic) tanpa menyimpan Bounding Box.
        """
        # --- PERUBAHAN 1: Hapus 'bboxes' dari deklarasi output ---
        data = {"id_number": None, "id_valid": False, "full_name": "Unknown Name", "dob": None}
        
        clean_blocks = []
        
        # 1. Filter Confidence
        for block in structured_ocr_output:
            # Tidak lagi menyimpan block['box']
            if block['conf'] >= self.CONF_THRESHOLD: 
                clean_blocks.append(block)

        # 2. LOGIKA PARSING NAMA (Adjacency)
        primer, segundo, nombre = "", "", ""
        
        for i, block in enumerate(clean_blocks):
            text = block['text'].upper()
            
            if ("PRIMER APELLIDO" in text or "PRIMER APELIIDO" in text):
                if i + 1 < len(clean_blocks): primer = clean_blocks[i+1]['text']
            
            elif "SEGUNDO APELLIDO" in text:
                if i + 1 < len(clean_blocks): segundo = clean_blocks[i+1]['text']
                
            elif "NOMBRE" in text and "PADRE" not in text:
                if i + 1 < len(clean_blocks): nombre = clean_blocks[i+1]['text']
        
        # 3. Cleanup & Gabung Nama (Sama)
        def clean_name(n):
            return re.sub(r'[^A-Z\s]', '', n.upper()).strip()

        # Urutan: [Nama Depan] [Apellido 1] [Apellido 2]
        full_name_list = [clean_name(nombre), clean_name(primer), clean_name(segundo)]
        data['full_name'] = " ".join([x for x in full_name_list if x and len(x) > 1])

        # 4. ID NUMBER (FINAL FIX REGEX)
        full_text = " ".join([block['text'] for block in clean_blocks])
        
        # Bersihkan noise umum (DNI NUM)
        cleaned_search_text = full_text.replace('NUM', '').replace('NÚM', '').replace('DNI', '').replace('ID', '').strip()
        
        id_match = re.search(r'([0-9]{8}[A-Z])', cleaned_search_text) 
        
        if id_match:
            data['id_number'] = id_match.group(1)
            data['id_valid'] = self.validate_dni_checksum(data['id_number'])
        
        # 5. TANGGAL LAHIR
        dob_match = re.search(r'(\d{2})[ ./-](\d{2})[ ./-](\d{4})', full_text)
        if dob_match:
            data['dob'] = f"{dob_match.group(1)}-{dob_match.group(2)}-{dob_match.group(3)}"

        return data