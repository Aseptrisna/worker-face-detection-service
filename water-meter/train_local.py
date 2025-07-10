from ultralytics import YOLO

def latih_model_lokal():
    """
    Fungsi untuk memulai proses pelatihan model YOLOv8 secara lokal.
    """
    # Muat model dasar YOLOv8 (yolov8n.pt adalah yang terkecil)
    model = YOLO('yolov8n.pt')

    # Mulai pelatihan dengan menunjuk ke file data.yaml Anda
    try:
        print("Memulai proses pelatihan model...")
        results = model.train(
            data='data.yaml',  # GANTI DENGAN PATH YAML ANDA
            epochs=75,          # Jumlah iterasi pelatihan (bisa ditambah jika perlu)
            imgsz=640,          # Ukuran gambar untuk pelatihan
            patience=15         # Berhenti jika tidak ada kemajuan setelah 15 epoch
        )
        print("✅ Pelatihan selesai!")
        print("   -> Model terbaik Anda disimpan di folder 'runs/detect/train/weights/best.pt'")
    except Exception as e:
        print(f"❌ Terjadi error saat pelatihan: {e}")

if __name__ == '__main__':
    # Pastikan Anda menjalankan skrip ini dari terminal
    # dengan virtual environment yang sudah aktif.
    latih_model_lokal()