import os
import cv2
import easyocr
import numpy as np
from pymongo import MongoClient
from ftplib import FTP
from dotenv import load_dotenv
import logging
import datetime
import io
import uuid

# ==============================================================================
# 1. KONFIGURASI DAN INISIALISASI
# ==============================================================================

# Muat variabel dari file .env
load_dotenv()

# Konfigurasi logging untuk memantau proses
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

# Konfigurasi dari .env
DEVICE_ID = os.getenv("DEVICE_ID", "CAM-P006")
FOLDER_GAMBAR_INPUT = os.getenv("FOLDER_GAMBAR_INPUT", "gambar_input")
os.makedirs(FOLDER_GAMBAR_INPUT, exist_ok=True)

MONGO_URI = os.getenv("DATABASE_LSKK")
DB_NAME = "hidroponik-sg"
COLLECTION_NAME = "reports" # Menggunakan nama koleksi 'reports' sesuai contoh

FTP_HOST = os.getenv("FTP_HOST_LSKK")
FTP_PORT = int(os.getenv("FTP_PORT_LSKK", 21))
FTP_USER = os.getenv("FTP_USER_LSKK")
FTP_PASS = os.getenv("FTP_PASS_LSKK")
FTP_FOLDER = os.getenv("FTP_FOLDER_LSKK", "/ocr_results/")
FTP_BASE_URL = os.getenv("FTP_BASE_URL")

# Konfigurasi OCR & ROI
ROI_AWAL = (650, 490, 120, 300)

# Inisialisasi OCR Reader
try:
    logging.info("[ℹ️] Memuat model EasyOCR ke memori...")
    reader = easyocr.Reader(['en'])
    logging.info("[✅] EasyOCR Reader berhasil dimuat.")
except Exception as e:
    logging.error(f"[❌] Gagal memuat EasyOCR Reader: {e}")
    exit(1)

# Inisialisasi Koneksi MongoDB
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
    mongo_client.server_info()
    logging.info("[✅] Koneksi ke MongoDB berhasil.")
except Exception as e:
    logging.error(f"[❌] Koneksi ke MongoDB gagal: {e}")
    exit(1)

# ==============================================================================
# 2. DEFINISI FUNGSI-FUNGSI UTAMA
# ==============================================================================

def upload_to_ftp_from_memory(image_bytes, remote_filename):
    """Mengunggah byte gambar dari memori ke server FTP."""
    try:
        with FTP() as ftp:
            ftp.connect(FTP_HOST, FTP_PORT)
            ftp.login(FTP_USER, FTP_PASS)
            
            try:
                ftp.cwd(FTP_FOLDER)
            except Exception:
                logging.warning(f"[⚠️] Folder FTP '{FTP_FOLDER}' tidak ditemukan, mencoba membuatnya.")
                ftp.mkd(FTP_FOLDER)
                ftp.cwd(FTP_FOLDER)
            
            with io.BytesIO(image_bytes) as image_file:
                ftp.storbinary(f"STOR {remote_filename}", image_file)
            
            logging.info(f"[✅] File '{remote_filename}' berhasil diunggah ke FTP.")
            return True
    except Exception as e:
        logging.error(f"[❌] Gagal mengunggah file ke FTP: {e}")
        return False

def save_to_mongodb(data_to_save):
    """Menyimpan data (dictionary) ke MongoDB."""
    try:
        insert_result = collection.insert_one(data_to_save)
        logging.info(f"[✅] Data dengan guid {data_to_save['guid']} berhasil disimpan ke MongoDB.")
        return True
    except Exception as e:
        logging.error(f"[❌] Gagal menyimpan data ke MongoDB: {e}")
        return False

def rotasi_dan_deteksi_ocr(path_gambar, roi_awal, ocr_reader):
    """Memutar gambar, mendeteksi angka, dan mengembalikan hasilnya."""
    try:
        img = cv2.imread(path_gambar)
        if img is None:
            logging.error(f"File gambar tidak dapat dibaca: {path_gambar}")
            return None, "Error: File tidak bisa dibaca"

        original_height, _, _ = img.shape
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        x_lama, y_lama, w_lama, h_lama = roi_awal
        x_baru = original_height - (y_lama + h_lama)
        y_baru = x_lama
        w_baru = h_lama
        h_baru = w_lama
        
        cropped_rotated_img = rotated_img[y_baru : y_baru + h_baru, x_baru : x_baru + w_baru]
        if cropped_rotated_img.size == 0:
            return None, "Error: ROI menghasilkan gambar kosong"

        result = ocr_reader.readtext(cropped_rotated_img, allowlist='0123456789', detail=0)
        angka_terdeteksi = ''.join(result) if result else "Gagal Deteksi"

        output_img = rotated_img.copy()
        cv2.rectangle(output_img, (x_baru, y_baru), (x_baru + w_baru, y_baru + h_baru), (0, 255, 0), 2)
        cv2.putText(output_img, angka_terdeteksi, (x_baru, y_baru - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return output_img, angka_terdeteksi

    except Exception as e:
        logging.error(f"Error saat memproses {os.path.basename(path_gambar)}: {e}")
        return None, f"Error: {e}"

# ==============================================================================
# 3. PROSES UTAMA
# ==============================================================================

def main():
    """Fungsi utama untuk menjalankan seluruh alur kerja."""
    logging.info(f"🚀 Memulai proses, membaca file dari folder: '{FOLDER_GAMBAR_INPUT}'")
    
    try:
        files_to_process = [f for f in os.listdir(FOLDER_GAMBAR_INPUT) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    except FileNotFoundError:
        logging.error(f"[❌] Folder input '{FOLDER_GAMBAR_INPUT}' tidak ditemukan.")
        return

    if not files_to_process:
        logging.warning("[⚠️] Tidak ada file gambar yang ditemukan di folder input.")
        return
        
    logging.info(f"Ditemukan {len(files_to_process)} gambar untuk diproses.")

    for filename in files_to_process:
        full_path = os.path.join(FOLDER_GAMBAR_INPUT, filename)
        logging.info(f"--- Memproses file: {filename} ---")
        
        processed_image, detected_number = rotasi_dan_deteksi_ocr(full_path, ROI_AWAL, reader)
        
        if processed_image is None or "Gagal" in detected_number or "Error" in detected_number:
            logging.error(f"Gagal memproses OCR untuk {filename}. Status: {detected_number}. Melewati file ini.")
            continue
        
        logging.info(f"Deteksi berhasil untuk {filename}. Angka terdeteksi: {detected_number}")
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        remote_filename = f"HASIL_OCR_{timestamp_str}_{filename}"
        
        is_success, buffer = cv2.imencode(".jpg", processed_image)
        if not is_success:
            logging.error(f"Gagal meng-encode gambar {filename} ke format JPG. Melewati file ini.")
            continue
            
        if upload_to_ftp_from_memory(buffer.tobytes(), remote_filename):
            # Normalisasi path untuk URL
            ftp_path_for_url = os.path.join(FTP_FOLDER, remote_filename).replace("\\", "/").lstrip("/")
            image_url = f"{FTP_BASE_URL}{remote_filename}"
            
            processing_time = datetime.datetime.now(datetime.timezone.utc)
            
            # ⭐️ Membuat struktur data MongoDB sesuai permintaan
            mongo_data = {
                "reporterName": f"AI-{DEVICE_ID}",
                "guid": str(uuid.uuid4()),
                "companyGuid": "COMPANY-0a3a303e-7dd0-4246-ad21-d675e77904b4-2024",
                "reportType": "AI-WaterMeterOCR",
                "guidDevice": DEVICE_ID,
                "reportContent": f"Deteksi angka meteran air - {DEVICE_ID}",
                "imageUrl": image_url,
                "detections": { # Adaptasi untuk hasil OCR
                    "type": "OCR",
                    "value": detected_number,
                    "roi_on_original_image": list(ROI_AWAL)
                },
                "num_faces": 0, # Tidak relevan untuk OCR
                "counts": {},   # Tidak relevan untuk OCR
                "latitude": 0.0,
                "longitude": 0.0,
                "timestamp": int(processing_time.timestamp()),
                "datetime": processing_time.isoformat(),
                "file_name": remote_filename,
                "date": processing_time,
                "createdAt": processing_time,
                "updatedAt": processing_time,
                "model": "EasyOCR",
                "version": "1.2"
            }
            print(mongo_data)
            save_to_mongodb(mongo_data)
        else:
            logging.error(f"Gagal mengunggah {filename} ke FTP, data tidak akan disimpan ke MongoDB.")

        logging.info(f"--- Selesai memproses file: {filename} ---\n")

    logging.info("✅ Semua file telah selesai diproses.")


if __name__ == "__main__":
    main()