# AI Image Processor

A service that processes images using AI (face detection) and integrates with RabbitMQ, MongoDB, and FTP.

## Features
- Consumes messages from RabbitMQ.
- Downloads images from a remote server.
- Detects faces in images using OpenCV.
- Saves processed images to FTP.
- Stores metadata in MongoDB.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-image-processor.git