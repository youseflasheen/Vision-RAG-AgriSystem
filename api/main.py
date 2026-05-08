from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from src.vision.inference import AgriVisionRAG

app = FastAPI(
    title="Vision-RAG Agricultural System API",
    description="An end-to-end AI system for crop disease detection and expert treatment advice.",
    version="1.0.0"
)

# Initialize the pipeline globally
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    # The pipeline will load the model and RAG DB when the server starts
    pipeline = AgriVisionRAG()

@app.post("/diagnose/")
async def diagnose_crop(file: UploadFile = File(...)):
    if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    # Save the uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run the full pipeline
        result = pipeline.predict_and_advise(temp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
async def root():
    return {"message": "Vision-RAG API is running."}