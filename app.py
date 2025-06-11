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
from ultralytics import YOLO

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

# Load YOLOv8 model (COCO pretrained model detects 80 classes including people, animals, vehicles)
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
    
    # COCO class names (80 classes)
    class_names = model.names
    logging.info(f"[ℹ️] Model can detect: {', '.join(class_names.values())}")
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
    """Mendeteksi objek dalam gambar menggunakan YOLOv8"""
    try:
        # Preprocess image
        height, width = image.shape[:2]
        max_dim = 1280
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
            logging.info(f"[ℹ️] Gambar di-resize ke {image.shape[1]}x{image.shape[0]}")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv8 inference
        results = model.predict(image_rgb, conf=0.5)
        
        # Initialize counters
        detections = {
            'person': [],
            'cat': [],
            'dog': [],
            'vehicle': [],  # Includes car, truck, bus, motorcycle, etc.
            'other_animals': []
        }
        
        # COCO class IDs for our categories
        class_ids = {
            'person': 0,
            'cat': 15,
            'dog': 16,
            'car': 2,
            'motorcycle': 3,
            'bus': 5,
            'truck': 7
        }
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = class_names[class_id]
                confidence = float(box.conf)
                
                # Categorize detections
                if class_id == class_ids['person']:
                    detections['person'].append({'class': class_name, 'confidence': confidence, 'bbox': box.xyxy[0].cpu().numpy().tolist()})
                elif class_id == class_ids['cat']:
                    detections['cat'].append({'class': class_name, 'confidence': confidence, 'bbox': box.xyxy[0].cpu().numpy().tolist()})
                elif class_id == class_ids['dog']:
                    detections['dog'].append({'class': class_name, 'confidence': confidence, 'bbox': box.xyxy[0].cpu().numpy().tolist()})
                elif class_id in [class_ids['car'], class_ids['motorcycle'], class_ids['bus'], class_ids['truck']]:
                    detections['vehicle'].append({'class': class_name, 'confidence': confidence, 'bbox': box.xyxy[0].cpu().numpy().tolist()})
                elif class_name in ['bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
                    detections['other_animals'].append({'class': class_name, 'confidence': confidence, 'bbox': box.xyxy[0].cpu().numpy().tolist()})
        
        logging.info(f"[✅] Deteksi objek: "
                    f"Manusia: {len(detections['person'])}, "
                    f"Kucing: {len(detections['cat'])}, "
                    f"Anjing: {len(detections['dog'])}, "
                    f"Kendaraan: {len(detections['vehicle'])}, "
                    f"Hewan lain: {len(detections['other_animals'])}")
        
        return detections, results[0].plot()  # Return detections and annotated image
    except Exception as e:
        logging.error(f"[❌] Error saat memproses deteksi objek: {e}")
        return {}, image

def upload_to_ftp(local_filepath, remote_filename):
    """Mengunggah file ke FTP server"""
    try:
        with FTP() as ftp:
            ftp.connect(FTP_HOST, FTP_PORT)
            ftp.login(FTP_USER, FTP_PASS)
            ftp.cwd(FTP_FOLDER)
            
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

def draw_detections_on_image(image, detections):
    """Menampilkan deteksi objek pada gambar"""
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Add summary text
        summary_text = (
            f"Manusia: {len(detections['person'])} | "
            f"Kucing: {len(detections['cat'])} | "
            f"Kendaraan: {len(detections['vehicle'])}"
        )
        cv2.putText(image, summary_text, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add class-specific colors
        colors = {
            'person': (0, 255, 0),     # Green
            'cat': (255, 0, 0),        # Blue
            'dog': (0, 0, 255),        # Red
            'vehicle': (255, 255, 0),  # Cyan
            'other_animals': (0, 255, 255)  # Yellow
        }
        
        # Draw all detections
        for category in detections:
            for detection in detections[category]:
                class_name = detection['class']
                confidence = detection['confidence']
                bbox = detection['bbox']
                
                # Get color based on class
                color = colors.get(category, (255, 255, 255))  # Default white
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(image, label, (x1, y1-10), font, 0.5, color, 1, cv2.LINE_AA)
        
        logging.info("[✅] Berhasil menambahkan deteksi pada gambar.")
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

        if upload_to_ftp(local_path, filename):
            delete_local_file(local_path)
            return filename
        else:
            logging.warning(f"[⚠️] Gagal mengunggah file ke FTP, file lokal tetap disimpan.")
            return None
    except Exception as e:
        logging.error(f"[❌] Error saat menyimpan gambar: {e}")
        return None

def callback(ch, method, properties, body):
    """Fungsi untuk menangani pesan dari RabbitMQ"""
    try:
        data = json.loads(body)
        logging.info(f"[📩] Menerima pesan: {data}")

        guid = data.get("guid")
        guid_device = data.get("guid_device")
        image_name = data.get("value")
        timestamp = data.get("timestamp")
        datetime_str = data.get("datetime")

        # Download gambar
        image = download_image(image_name)
        if image is None:
            logging.error(f"[❌] Gagal memproses pesan {guid}: gambar tidak ditemukan.")
            return

        # Proses deteksi objek
        detections, annotated_image = process_image(image)

        # Tandai gambar dengan deteksi
        marked_image = draw_detections_on_image(annotated_image, detections)

        # Simpan gambar hasil AI
        ai_image_filename = f"AI_{image_name}"
        saved_filename = save_image(marked_image, ai_image_filename)
        if saved_filename is None:
            logging.error(f"[❌] Gagal menyimpan gambar AI untuk {guid}.")
            return

        # Insert data ke MongoDB
        new_data = {
            "name": "AI",
            "guid": guid,
            "guid_device": guid_device,
            "image_name": image_name,
            "ai_image_name": saved_filename,
            "detections": detections,
            "num_faces": len(detections['person']),
            "counts": {
                "person": len(detections['person']),
                "cat": len(detections['cat']),
                "dog": len(detections['dog']),
                "vehicle": len(detections['vehicle']),
                "other_animals": len(detections['other_animals'])
            },
            "timestamp": timestamp,
            "datetime": datetime_str,
            "file_name": saved_filename,
            "createdAt": datetime.datetime.now(),
            "updatedAt": datetime.datetime.now(),
            "model": "YOLOv8n",
            "version": "1.1"
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