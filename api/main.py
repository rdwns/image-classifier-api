from fastapi import FastAPI, File, UploadFile, HTTPException, status
from config import ClassifierResponse
from classifier import classify
from PIL import Image
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


#The /predict API endpoint that can be used for inference

@app.post("/predict", response_model=ClassifierResponse)
async def predict(file: UploadFile = File(...)):

    #We're rejecting all filetypes other than images
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File {file.filename} is not an image!')

    try:

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB') #Create file-like object in memory to save image without writing to disk

        predicted_class = image_classifier.predict(image)
        logging.info(f'Predicted Class: {predicted_class}')

        return {
            "filename": file.filename,
            "contentype": file.content_type,
            "predicted_class": predicted_class['class'],
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))
