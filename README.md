# Predictive Medical Equipment Maintenance System (Streamlit UI)

## Overview
This Streamlit application provides a user-friendly interface to predict equipment failures based on sensor data. Users can input equipment parameters and receive insights into equipment health, maintenance levels, and associated risks.

---

## Features
- **Interactive Interface:** Simple forms to input equipment parameters.
- **Real-time Visualization:** Displays equipment health and maintenance levels.
- **Color-Coded Status:** Equipment health and maintenance levels are color-coded (Green, Yellow, Red) for easy interpretation.

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- Streamlit
- Pandas
- Joblib

### Installation
```bash
# Clone the repository
git clone https://github.com/Swarno-Coder/Predictive-Medical-Equipment-Maintenance-System.git
cd pmems

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Running the Application
```bash
streamlit run app.py
```
- The application will be available at `http://localhost:8501`

---

## Usage
1. **Input Parameters:**
   - Enter equipment details in the provided fields.
   - Click the `Predict` button.
2. **View Results:**
   - Equipment health and maintenance levels will be displayed side by side.
   - Results are color-coded:
     - **Green:** Good health, low maintenance.
     - **Yellow:** Moderate health, medium maintenance.
     - **Red:** Critical health, high maintenance.

---

## Model Training and Dataset Generation
1. **Dataset Generation:**
   - Use synthetic sensor data or gather real-world data.
   - Label data based on failure occurrences.
   or run the `data_generation.py` script to generate synthetic data.
   ```bash
    python data_generation.py
    ```
2. **Model Training:**
   - Use a `RandomForestClassifier` or any regression model.
   - Train the model with features like `temperature`, `humidity`, and `vibration`.
   - Save the model as `equipment_model.pkl` using `joblib.dump(model, 'equipment_model.pkl')`.

   or run the `model_training.py` script to train a sample model.
   ```bash
    python model_training.py
    ```

---

## Troubleshooting
- **UI Not Loading:** Ensure Streamlit is installed and the correct Python environment is activated.
- **Incorrect Predictions:** Retrain the model with updated data.

---

## License
This project is licensed under the Apache License.
