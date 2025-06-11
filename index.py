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
from ultralytics import YOLO  # For YOLOv8 model

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
FTP_PORT = int(os.getenv("FTP_PORT", 21))
FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")
FTP_FOLDER = os.getenv("FTP_FOLDER", "/")

# Load YOLOv8 model (download if not exists)
MODEL_PATH = "yolov8n.pt"
if not os.path.exists(MODEL_PATH):
    logging.info("[ℹ️] Downloading YOLOv8 model...")
    try:
        model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        response = requests.get(model_url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
    except Exception as e:
        logging.error(f"[❌] Failed to download YOLOv8 model: {e}")
        exit(1)

# Load YOLOv8 model
try:
    model = YOLO(MODEL_PATH)
    logging.info("[✅] YOLOv8 model loaded successfully.")
except Exception as e:
    logging.error(f"[❌] Failed to load YOLOv8 model: {e}")
    exit(1)

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
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            # Check if image is valid
            if image is None or image.size == 0:
                logging.error(f"[❌] Gambar {image_name} tidak valid setelah diunduh.")
                return None
                
            logging.info(f"[✅] Gambar {image_name} berhasil diunduh. Ukuran: {image.shape}")
            return image
        else:
            logging.error(f"[❌] Gagal mengunduh gambar {image_name}, status code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"[❌] Error saat mengunduh gambar {image_name}: {e}")
        return None

def process_image(image):
    """Mendeteksi jumlah manusia dalam gambar menggunakan YOLOv8"""
    try:
        # Preprocess image - resize if too large
        height, width = image.shape[:2]
        max_dim = 1280
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
            logging.info(f"[ℹ️] Gambar di-resize ke {image.shape[1]}x{image.shape[0]} untuk pemrosesan lebih cepat")

        # Convert to RGB (YOLO expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv8 inference
        results = model.predict(image_rgb, conf=0.5)  # Confidence threshold 50%
        
        # Filter only person detections (class 0 in COCO dataset)
        person_detections = []
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:  # Class 0 is person in COCO
                    person_detections.append(box)
        
        num_persons = len(person_detections)
        logging.info(f"[✅] Jumlah manusia terdeteksi: {num_persons} dengan confidence {[round(float(d.conf), 2) for d in person_detections]}")
        
        return num_persons, person_detections, results[0].plot()  # Return annotated image
    except Exception as e:
        logging.error(f"[❌] Error saat memproses deteksi manusia: {e}")
        return 0, [], image

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

def draw_detections_on_image(image, num_persons, detections):
    """Menampilkan jumlah manusia yang terdeteksi pada gambar dengan bounding boxes"""
    try:
        # The annotated image is already returned from YOLO's plot() function
        # We'll just add the count text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Jumlah manusia: {num_persons}"
        cv2.putText(image, text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add confidence for each detection
        for i, det in enumerate(detections):
            conf = float(det.conf)
            box = det.xyxy[0].cpu().numpy()
            cv2.putText(image, f"{conf:.2f}", (int(box[0]), int(box[1])-10), 
                        font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        logging.info(f"[✅] Berhasil menambahkan deteksi pada gambar.")
        return image
    except Exception as e:
        logging.error(f"[❌] Error saat menandai gambar: {e}")
        return image

def save_image(image, filename):
    """Menyimpan gambar hasil AI"""
    try:
        local_path = os.path.join(SAVE_FOLDER, filename)
        success = cv2.imwrite(local_path, image)
        if not success:
            logging.error(f"[❌] Gagal menyimpan gambar ke {local_path}")
            return None
            
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

        # 2️⃣ Proses deteksi manusia dengan YOLOv8
        num_persons, detections, annotated_image = process_image(image)

        # 3️⃣ Tandai gambar dengan deteksi
        marked_image = draw_detections_on_image(annotated_image, num_persons, detections)

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
            "num_persons": num_persons,
            "detections": [{
                "confidence": float(det.conf),
                "bbox": det.xyxy[0].cpu().numpy().tolist()
            } for det in detections],
            "timestamp": timestamp,
            "datetime": datetime_str,
            "file_name": saved_filename,
            "createdAt": datetime.datetime.now(),
            "updatedAt": datetime.datetime.now(),
            "model": "YOLOv8n",
            "version": "1.0"
        }
        
        insert_result = collection.insert_one(new_data)

        if insert_result.inserted_id:
            logging.info(f"[✅] Data dengan guid {guid} berhasil disimpan di MongoDB.")
        else:
            logging.warning(f"[⚠️] Gagal menyimpan data dengan guid {guid}.")

    except json.JSONDecodeError as e:
        logging.error(f"[❌] Error decoding JSON message: {e}")
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