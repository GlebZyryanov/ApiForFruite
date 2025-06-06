import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def load_model_and_classes(model_path: str, classes_path: str):
    """Загружает модель и список классов"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Файл классов не найден: {classes_path}")
    
    model = load_model(model_path)
    with open(classes_path, 'r') as f:
        class_names = json.load(f)
    
    return model, class_names

def predict_image(model, image_path: str, class_names: list, img_size=(224, 224)):
    """Выполняет предсказание на изображении"""
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions, axis=1)[0]
    
    return class_names[predicted_idx]