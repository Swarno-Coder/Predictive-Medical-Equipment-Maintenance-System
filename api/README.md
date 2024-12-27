# Predictive Medical Equipment Maintenance System API (using FastAPI)

## Overview
This project provides a **Predictive Medical Equipment Maintenance System API** using **FastAPI**. The API predicts equipment failure probability based on sensor data and environmental parameters, offering insights into equipment health and required maintenance levels.

---

## Features
- **Prediction Endpoint:** Submit equipment data to receive failure probability, health status, and maintenance recommendations.
- **Health Check:** Simple endpoint to verify if the API is running.
- **Model Integration:** Uses a pre-trained model set(`equipment_model{i}.pkl`) for predictions.

DISCLAIMER: The model files must be named `equipment_model{i}.pkl`, where `i` is the model number. The API script will load the model files based on the number of models available. 

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- FastAPI
- Uvicorn
- Joblib
- Pandas

### Installation
```bash
# Clone the repository
git clone https://github.com/Swarno-Coder/Predictive-Medical-Equipment-Maintenance-System.git pmems
cd pmems/api

# Create a virtual environment
python -m venv nenv
source nenv/bin/activate  # Windows: nenv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Running the API
```bash
uvicorn pmems_api:app --reload
```

- The API will be available at `http://127.0.0.1:8000`

---

## Usage

### Health Check
```bash
curl -X GET http://127.0.0.1:8000/health
```
**Response:**
```json
{
  "status": "API is running"
}
```

### Prediction Endpoint
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d @test_payload.json
```
**Example Payload (test_payload.json):**
```json
{
  "equipment_type": "Ventilator",
  "temperature_avg": 45.6,
  "humidity_avg": 70.2,
  "last_maintenance_days": 50,
  "uptime_hours": 4000,
  "sensor_1": 12.3,
  "sensor_2": 5.8,
  "vibration": 0.9,
  "voltage_fluctuation": 1.4,
  "pressure": 15.5
}
```

**Example Response:**
```json
{
  "equipment_health": "Moderate",
  "maintenance_level": "Medium",
  "failure_probability": 0.55,
  "failure_reason": "Wear and tear",
  "cost_implications": 1500.0,
  "updated_uptime": 4200
}
```

---

## Notes
- The API uses a synthetic model trained on dummy data.
- The model file set `equipment_model{i}.pkl` must be in the same directory as the API script.
- Adjust `equipment_model{i}.pkl` based on updated datasets for more accurate predictions.

---

## Troubleshooting
- **500 Internal Server Error:** Ensure that `equipment_model{i}.pkl` exists and is compatible with the API.
- **Invalid Payloads:** Ensure all required fields are included and match the expected data types.

---

## License
This project is licensed under the Apache License.


