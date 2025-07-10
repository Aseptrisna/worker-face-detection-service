import cv2
import easyocr
import numpy as np

def deteksi_langsung_pada_roi(path_gambar, roi, path_simpan):
    """
    Fungsi untuk langsung mendeteksi angka pada ROI di gambar asli.
    1. Membaca gambar asli.
    2. Memotong area ROI.
    3. Mendeteksi angka di dalam area ROI.
    4. Menggambar kotak dan hasil deteksi pada gambar asli.
    """
    try:
        # 1. Baca gambar asli
        img = cv2.imread(path_gambar)
        if img is None:
            return "Error: File gambar tidak dapat dibaca.", None

        # 2. Potong (crop) area ROI langsung dari gambar asli
        x, y, w, h = roi
        cropped_img = img[y : y + h, x : x + w]

        # 3. Deteksi angka di dalam area yang sudah dipotong
        # Inisialisasi reader di dalam fungsi jika jarang dipanggil.
        # Untuk pemrosesan batch, lebih baik inisialisasi sekali di luar.
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_img, allowlist='0123456789.', detail=0)
        
        # Gabungkan hasil deteksi menjadi satu string angka
        angka_terdeteksi = ''.join(result) if result else "Gagal Deteksi"

        # 4. Gambar kotak dan teks hasil pada gambar ASLI
        output_img = img.copy()
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_img, angka_terdeteksi, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 5. Simpan gambar hasil
        cv2.imwrite(path_simpan, output_img)

        return angka_terdeteksi, path_simpan

    except Exception as e:
        return f"Terjadi error: {e}", None

# --- EKSEKUSI KODE ---

# Path ke gambar meteran Anda
nama_file_gambar = 'E:/DATA MONJA/Data Power Cam/POWERCAM-P001-00eIacgG1.jpg'

# Nama untuk file output yang akan disimpan
path_hasil = 'Test-KWH.jpg'

# Tentukan ROI pada gambar ASLI
# Format: (koordinat x, koordinat y, lebar, tinggi)
roi_kwh = (150, 480, 1400, 300) 

# Jalankan semua proses
hasil_angka, file_tersimpan = deteksi_langsung_pada_roi(nama_file_gambar, roi_kwh, path_hasil)

# Cetak laporan hasil
if file_tersimpan and "Gagal" not in hasil_angka:
    print(f"✅ Deteksi berhasil!")
    print(f"   -> Angka Terdeteksi: {hasil_angka}")
    print(f"   -> Gambar hasil disimpan di: {file_tersimpan}")
else:
    print(f"❌ Deteksi gagal.")
    print(f"   -> Info: {hasil_angka}")
    if file_tersimpan:
        print(f"   -> Gambar hasil disimpan di: {file_tersimpan}")