from fastapi import FastAPI, File, UploadFile, HTTPException, status
from config import ClassifierResponse
from classifier import classify
from PIL import Image
import mxnet as mx
import numpy as np
import logging
import time
import sys
import io


app = FastAPI()


#Setup basic logging to print out logs to console
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

#Load the Classifier
image_classifier = classify.Classifier()


#Setting up a basic endpoint to perform healthchecks

@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    current_time = time.strftime('%H:%M%p %Z on %b %d, %Y')
    logging.info(f"API up and running!")
    return {'ping': 'pong'}


#The API endpoint that can be used for inference

@app.post("/predict", response_model=ClassifierResponse)
async def predict(file: UploadFile = File(...)):

    #We're rejecting all filetypes other than images
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File {file.filename} is not an image!')

    try:

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB') #Create file-like object in memory to save image without writing to disk

        #Need to convert the uploaded image to a mxnet compatible NDArray
        nd_img = mx.ndarray.array(image)

        predicted_class = image_classifier.predict(nd_img)
        logging.info(f'Predicted Class: {predicted_class}')

        return {
            "filename": file.filename,
            "contentype": file.content_type,
            "predicted_class": predicted_class['class'],
            "probability": predicted_class['probability']
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))
