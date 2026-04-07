# 🚗 Dr Drive – AI Car Inspection System

Dr Drive is an AI-powered Android application that analyzes used cars using OBD data and computer vision to detect issues, predict failures, estimate repair costs, and calculate fair resale value.

---

## 📱 Features

* 🔌 OBD-II Bluetooth Integration (ELM327)
* 📸 Image-based Damage Detection (YOLOv8)
* 🧠 AI-based Health Scoring
* 🔮 Failure Prediction
* 💰 Repair Cost Estimation (India)
* 📉 Car Value Calculator
* ⚠️ Fraud Detection

---

## 🧩 Tech Stack

### Android

* Java + XML
* MVVM Architecture
* Retrofit (API calls)

### Backend

* FastAPI
* Python

### AI/ML

* XGBoost / Random Forest
* YOLOv8 (Computer Vision)
* OpenCV

---

## 🏗️ Architecture

```
Car → OBD Device → Android App → Backend → ML Models → Results
```

---

## 📂 Project Structure

* `android-app/` → Android application
* `backend/` → FastAPI backend
* `ml-models/` → ML training & models

---

## 🚀 Setup Instructions

### Android App

1. Open in Android Studio
2. Connect device/emulator
3. Run the app

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 📸 Screenshots

(Add screenshots here)

---

## 🔮 Future Improvements

* Real-time streaming OBD data
* Better damage detection model
* India-specific pricing dataset
* Cloud deployment

---

## 🤝 Contributing

Pull requests are welcome.

---

## 📜 License

MIT License
