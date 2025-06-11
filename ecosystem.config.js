module.exports = {
    apps: [
      {
        name: "ai-image-processor", // Nama aplikasi
        script: "python3",          // Perintah untuk menjalankan Python
        args: "index.py",            // File Python yang akan dijalankan
        interpreter: "",            // Biarkan kosong untuk menggunakan Python default
        env: {
          NODE_ENV: "production",
        },
        log_date_format: "YYYY-MM-DD HH:mm:ss",
        error_file: "logs/error.log", // File log untuk error
        out_file: "logs/out.log",     // File log untuk output
        merge_logs: true,
        autorestart: true,            // Restart otomatis jika crash
        watch: false,                 // Jangan pantau perubahan file
        max_memory_restart: "1G",     // Restart jika penggunaan memori melebihi 1GB
      },
    ],
  };