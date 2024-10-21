from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
import io
from PIL import Image
import numpy as np
import logging

app = FastAPI()
reader = easyocr.Reader(['en'])

logging.basicConfig(level=logging.INFO)

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    try:
        logging.info("Received file: %s", file.filename)

        image_bytes = await file.read()
        logging.info("Image bytes length: %d", len(image_bytes))

        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        logging.info("Image successfully opened and converted to RGB.")

        image_np = np.array(image)
        logging.info("Image successfully converted to NumPy array.")

        results = reader.readtext(image_np)
        logging.info("Text extraction completed.")

        extracted_text = ' '.join(result[1] for result in results)
        logging.info("Extracted text: %s", extracted_text)

        return JSONResponse(content={"text": extracted_text})

    except Exception as e:
        logging.error("Error processing the image: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
