import os
import json
import cv2
import requests
import numpy as np
import pika
from pymongo import MongoClient
from dotenv import load_dotenv
from ftplib import FTP
import logging
import datetime

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Muat konfigurasi dari .env
load_dotenv()

# Konfigurasi MongoDB
MONGO_URI = os.getenv("DATABASE")
DB_NAME = "hylab"
COLLECTION_NAME = "datahasils"

# Konfigurasi RabbitMQ
RMQ_HOST = os.getenv("RMQ_HOST")
RMQ_USER = os.getenv("RMQ_USER")
RMQ_PASS = os.getenv("RMQ_PASS")
RMQ_PORT = int(os.getenv("RMQ_PORT", 5672))
RMQ_VHOST = os.getenv("RMQ_VHOST")
QUEUE_NAME = "service.ai"

# Folder penyimpanan gambar
SAVE_FOLDER = "processed_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Konfigurasi FTP
FTP_HOST = os.getenv("FTP_HOST")
FTP_PORT = int(os.getenv("FTP_PORT", 21))  # Diperbaiki
FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")
FTP_FOLDER = os.getenv("FTP_FOLDER", "/")

# Koneksi ke MongoDB
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    logging.info("[✅] Koneksi ke MongoDB berhasil.")
except Exception as e:
    logging.error(f"[❌] Koneksi ke MongoDB gagal: {e}")
    exit(1)

# Koneksi ke RabbitMQ
try:
    credentials = pika.PlainCredentials(RMQ_USER, RMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RMQ_HOST,
        port=RMQ_PORT,
        virtual_host=RMQ_VHOST,
        credentials=credentials
    )
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True, passive=True)
    logging.info(f"[✅] Koneksi ke RabbitMQ berhasil, menunggu pesan di antrean '{QUEUE_NAME}'")
except Exception as e:
    logging.error(f"[❌] Gagal terhubung ke RabbitMQ: {e}")
    exit(1)



def download_image(image_name):
    """Mengunduh gambar dari server"""
    image_url = f"https://image-view.sta.my.id/data/{image_name}"
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            logging.info(f"[✅] Gambar {image_name} berhasil diunduh.")
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            logging.error(f"[❌] Gagal mengunduh gambar {image_name}, status code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"[❌] Error saat mengunduh gambar {image_name}: {e}")
        return None


def process_image(image):
    """Mendeteksi jumlah manusia dalam gambar menggunakan Haar Cascade"""
    try:
        # Load pre-trained Haar Cascade model untuk deteksi wajah
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Konversi gambar ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Hitung jumlah wajah yang terdeteksi
        num_faces = len(faces)
        
        logging.info(f"[✅] Jumlah manusia terdeteksi: {num_faces}")
        return num_faces
    except Exception as e:
        logging.error(f"[❌] Error saat memproses deteksi manusia: {e}")
        return 0


def upload_to_ftp(local_filepath, remote_filename):
    """Mengunggah file ke FTP server"""
    try:
        with FTP() as ftp:
            ftp.connect(FTP_HOST, FTP_PORT)
            ftp.login(FTP_USER, FTP_PASS)
            ftp.cwd(FTP_FOLDER)  # Pindah ke folder tujuan
            
            with open(local_filepath, "rb") as file:
                ftp.storbinary(f"STOR {remote_filename}", file)

            logging.info(f"[✅] File '{remote_filename}' berhasil diunggah ke FTP {FTP_FOLDER}")
            return True
    except Exception as e:
        logging.error(f"[❌] Gagal mengunggah file ke FTP: {e}")
        return False


def delete_local_file(filepath):
    """Menghapus file dari penyimpanan lokal"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f"[🗑] File lokal dihapus: {filepath}")
        else:
            logging.warning(f"[⚠️] File tidak ditemukan untuk dihapus: {filepath}")
    except Exception as e:
        logging.error(f"[❌] Error saat menghapus file lokal: {e}")


def draw_text_on_image(image, num_faces):
    """Menampilkan jumlah manusia yang terdeteksi pada gambar"""
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Jumlah manusia: {num_faces}"
        cv2.putText(image, text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        logging.info(f"[✅] Berhasil menambahkan teks '{text}' pada gambar.")
        return image
    except Exception as e:
        logging.error(f"[❌] Error saat menandai gambar: {e}")
        return image


def save_image(image, filename):
    """Menyimpan gambar hasil AI"""
    try:
        local_path = os.path.join(SAVE_FOLDER, filename)
        cv2.imwrite(local_path, image)
        logging.info(f"[✅] Gambar hasil diproses dan disimpan sebagai: {local_path}")

        # Upload ke FTP
        if upload_to_ftp(local_path, filename):
            # Hapus file lokal setelah sukses upload ke FTP
            delete_local_file(local_path)
            return filename  # Return nama file jika sukses
        else:
            logging.warning(f"[⚠️] Gagal mengunggah file ke FTP, file lokal tetap disimpan.")
            return None  # Return None jika gagal upload
    except Exception as e:
        logging.error(f"[❌] Error saat menyimpan gambar: {e}")
        return None


def callback(ch, method, properties, body):
    """Fungsi untuk menangani pesan dari RabbitMQ"""
    try:
        data = json.loads(body)
        logging.info(f"[📩] Menerima pesan: {data}")

        # Ekstrak data dari pesan
        guid = data.get("guid")
        guid_device = data.get("guid_device")
        image_name = data.get("value")  # Nama file gambar
        timestamp = data.get("timestamp")
        datetime_str = data.get("datetime")

        # 1️⃣ Download gambar
        image = download_image(image_name)
        if image is None:
            logging.error(f"[❌] Gagal memproses pesan {guid}: gambar tidak ditemukan.")
            return

        # 2️⃣ Proses deteksi manusia
        num_faces = process_image(image)

        # 3️⃣ Tandai gambar dengan jumlah manusia yang terdeteksi
        marked_image = draw_text_on_image(image, num_faces)

        # 4️⃣ Simpan gambar hasil AI dan unggah ke FTP
        ai_image_filename = f"AI_{image_name}"  # Nama file baru untuk hasil AI
        saved_filename = save_image(marked_image, ai_image_filename)
        if saved_filename is None:
            logging.error(f"[❌] Gagal menyimpan gambar AI untuk {guid}.")
            return

        # 5️⃣ Insert data baru ke MongoDB
        new_data = {
            "name": "AI",
            "guid": guid,
            "guid_device": guid_device,
            "image_name": image_name,
            "ai_image_name": saved_filename,
            "num_faces": num_faces,
            "timestamp": timestamp,
            "datetime": datetime_str,
            "file_name":saved_filename,
            "createdAt": datetime.datetime.now(),  # Tambahkan createdAt
            "updatedAt": datetime.datetime.now()   # Tambahkan updatedAt
        }
        insert_result = collection.insert_one(new_data)

        if insert_result.inserted_id:
            logging.info(f"[✅] Data dengan guid {guid} berhasil disimpan di MongoDB.")
        else:
            logging.warning(f"[⚠️] Gagal menyimpan data dengan guid {guid}.")

    except Exception as e:
        logging.error(f"[❌] Error dalam proses callback: {e}")


# Mulai konsumsi pesan dari RabbitMQ
logging.info("[🚀] Menunggu pesan dari RabbitMQ...")
try:
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()
except KeyboardInterrupt:
    logging.info("[🛑] Program dihentikan oleh pengguna.")
    connection.close()
except Exception as e:
    logging.error(f"[❌] Terjadi error pada RabbitMQ: {e}")
    connection.close()