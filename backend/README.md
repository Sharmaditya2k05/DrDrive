# Dr Drive — Backend API

AI-powered car inspection backend. FastAPI + SQLAlchemy + YOLOv8 + XGBoost.

## Quick Start (local dev — no Docker)

```bash
# 1. Clone / extract project
cd DrDriveBackend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy env file
cp .env.example .env
# Edit .env if needed (defaults work for dev with SQLite)

# 5. Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the Swagger UI.

---

## Quick Start (Docker)

```bash
docker-compose up --build
```

---

## Train ML Models

Place your CSV datasets in `data/`:

```bash
mkdir data
# Copy engine_failure_dataset.csv, engine_data.csv here

python scripts/train_fault_model.py
# → saves ml_models/fault_classifier.pkl
```

For YOLOv8 damage detection:

```bash
# Extract archive__6_ and archive__9_ into data/damage/
python scripts/train_yolo.py
# → saves ml_models/yolov8_damage.pt
```

Without model files, the API uses rule-based fallbacks automatically.

---

## Project Structure

```
DrDriveBackend/
├── app/
│   ├── main.py                    ← FastAPI app entry point
│   ├── core/
│   │   ├── config.py              ← All settings (pydantic-settings)
│   │   ├── security.py            ← JWT + bcrypt
│   │   └── dependencies.py        ← FastAPI DI (current user)
│   ├── db/
│   │   ├── session.py             ← Async SQLAlchemy engine
│   │   ├── base.py                ← DeclarativeBase
│   │   └── crud.py                ← All DB queries
│   ├── models/
│   │   ├── user.py                ← User ORM
│   │   └── inspection.py         ← Inspection ORM
│   ├── schemas/
│   │   ├── auth.py                ← Login/register schemas
│   │   └── inspection.py         ← Report schemas
│   ├── api/routes/
│   │   ├── auth.py                ← POST /api/auth/login|register
│   │   ├── inspection.py         ← POST/GET /api/inspection/
│   │   └── health.py             ← GET /health
│   └── services/
│       ├── inspection_service.py  ← Full analysis pipeline
│       ├── storage.py             ← S3 / local image storage
│       └── ml/
│           ├── fault_detector.py  ← XGBoost OBD faults
│           ├── damage_detector.py ← YOLOv8 damage
│           ├── failure_predictor.py ← Failure timeline
│           ├── health_scorer.py   ← 0-100 score
│           └── valuation_engine.py ← Indian market price
├── scripts/
│   ├── train_fault_model.py       ← Train XGBoost
│   └── train_yolo.py              ← Fine-tune YOLOv8
├── tests/
│   └── test_api.py               ← Full pytest suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET  | `/health` | No | Server health check |
| POST | `/api/auth/register` | No | Create account |
| POST | `/api/auth/login` | No | Get JWT token |
| POST | `/api/inspection/create` | Yes | Upload OBD + images |
| GET  | `/api/inspection/{id}` | Yes | Poll analysis result |
| GET  | `/api/inspection/list` | Yes | List all inspections |

---

## Android Emulator Connection

The Android emulator reaches your laptop's localhost via `10.0.2.2`.
The `BASE_URL` in the Android `build.gradle` debug config is already set to:
```
http://10.0.2.2:8000/
```
