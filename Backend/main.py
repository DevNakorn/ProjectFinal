from fastapi import FastAPI, UploadFile, Form
import joblib
import os
import shutil

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile, tone: str = Form(...)):
    #save ชั่วคราว
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    #load model
    model_path = f"models/{tone}.pkl"
    if not os.path.exists(model_path):
        return {"error" : f"Model for {tone} not found!"}
    
    model = joblib.load(model_path)

    # เช่น model.predict([feature]) -> [Exposure, Contrast]
    prediction = model.predict([temp_file])  
    exposure, contrast = prediction[0]

    # ลบไฟล์ temp
    os.remove(temp_file)

    return {"exposure": float(exposure), "contrast": float(contrast)}