from fastapi import FastAPI, HTTPException, Request
import pandas as pd
import joblib

app = FastAPI()
def load_model(): return [joblib.load('..\models\equipment_model.pkl'), joblib.load('..\models\equipment_model2.pkl'), joblib.load('..\models\equipment_model3.pkl'), joblib.load('..\models\equipment_model4.pkl')]  # Load your model file

model = load_model()

@app.post("/predict")
async def predict_equipment(request: Request):
    try:
        # Receive JSON data from frontend
        input_data = await request.json()
        input_df = pd.DataFrame([input_data])

        # Predict using the model
        prediction = [model[0].predict(input_df),
                    model[1].predict(input_df),
                    model[2].predict(input_df),
                    model[3].predict(input_df)]
        
        output = [pd.DataFrame(prediction[0], columns=['failure_probability']),
              pd.DataFrame(prediction[3], columns=['failure_reason']),
              pd.DataFrame(prediction[1], columns=['cost_implications']),
              pd.DataFrame(prediction[2], columns=['updated_uptime'])]
        
        failure_prob = float(output[0]['failure_probability'][0])

        if failure_prob < 0.3:
            health = 'Good'
            maintenance_level = 'Low'
        elif failure_prob < 0.6:
            health = 'Moderate'
            maintenance_level = 'Medium'
        else:
            health = 'Critical'
            maintenance_level = 'High'

        response = {
            'equipment_health': health,
            'maintenance_level': maintenance_level,
            'failure_probability': failure_prob,
            'failure_reason': str(output[1]['failure_reason'][0]),
            'cost_implications': float(output[2]['cost_implications'][0]),
            'updated_uptime': float(output[3]['updated_uptime'][0])
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "API is running"}