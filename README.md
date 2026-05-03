# 📘 AI-Based Attendance System

This project presents an **automated attendance system using face recognition and blink detection** for accurate and secure attendance marking.

Initially, student data is collected using a webcam. The system captures face images and stores them in a structured dataset along with roll numbers. These images are then used to train a face recognition model using the **LBPH (Local Binary Pattern Histogram) algorithm**.

During execution, the system uses a webcam to capture live video. It detects faces using OpenCV and identifies students by comparing them with the trained model. To ensure the presence of a real person, **blink detection (liveness verification)** is performed using MediaPipe by analyzing eye movements.

Once a student is successfully recognized and verified, attendance is automatically recorded with **name, roll number, date, and time** in a formatted Excel sheet.

---

## 🚀 Features

- ✅ Real-time face recognition  
- 👁️ Blink-based liveness detection  
- 📊 Automatic attendance marking  
- 📁 Excel report generation  
- 🆔 Roll number integration  

---

## ⚙️ Project Workflow

1. 📸 Collect student dataset (face images + roll number)  
2. 🧠 Train face recognition model  
3. 🎥 Run attendance system (face + blink detection)  

---

## ▶️ How to Run

Run the following commands in order:

```bash
python collect_dataset.py
python train_model.py
python attendance.py
