from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from .utils import load_model_and_classes, predict_image

app = FastAPI(
    title="Fruit Recognition API",
    description="API для классификации изображений фруктов",
    version="0.1.0"
)

# Загрузка модели и классов при старте
model, class_names = None, []

@app.on_event("startup")
async def startup_event():
    global model, class_names
    try:
        model, class_names = load_model_and_classes(
            model_path="../PE_fruitrecognition-main/fruit_recognition_model.h5",
            classes_path="../PE_fruitrecognition-main/class_names.json"
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

@app.post("/predict/", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="Неверный формат файла")
    
    try:
        # Сохраняем временный файл
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())
        
        # Предсказание
        prediction = predict_image(
            model=model,
            image_path=temp_file,
            class_names=class_names,
            img_size=(224, 224)
        )
        return {"class": prediction}
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)