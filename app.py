from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware


model = load_model("history.h5")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

def preprocess_image(image: Image.Image) -> np.ndarray:
    
    if image.mode != 'L':
        image = image.convert('L')
    
    
    image = image.resize((48, 48))
    
    
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0  
    img_array = np.expand_dims(img_array, axis=-1) 
    img_array = np.expand_dims(img_array, axis=0)   
    
    return img_array

@app.post("/predict-emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
   
    processed_image = preprocess_image(image)
    
   
    prediction = model.predict(processed_image)
    
   
    predicted_emotion = np.argmax(prediction, axis=1)[0]
    
  
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    
    return {"emotion": emotion_labels[predicted_emotion]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
