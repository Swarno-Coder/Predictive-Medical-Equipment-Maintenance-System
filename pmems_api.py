from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import uvicorn
import pandas as pd
import joblib
import json

app = FastAPI()

# Load the previously saved model
model = joblib.load('/path/to/your/saved_model.pkl')

class InputData(BaseModel):
    data: List[List[Union[str, float, int]]]

class OutputData(BaseModel):
    data: List[List[float]]

@app.post("/process-data", response_model=OutputData)
async def process_data(input_data: InputData):
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame(input_data.data)
    
    # Perform inference using the model
    predictions = model.predict(input_df)
    
    # Convert predictions to a list of lists
    processed_data = predictions.tolist()
    
    # Save the processed data to a JSON file
    with open('/path/to/output.json', 'w') as json_file:
        json.dump(processed_data, json_file)
    
    return OutputData(data=processed_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)