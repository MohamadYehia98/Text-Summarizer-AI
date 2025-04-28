# pip install fastapi uvicorn transformers sentencepiece
# pip install jinja2
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import re


import os
import subprocess
import zipfile

# Folder inside your workspace to store the model
MODEL_DIR = "Text_Summarizer"
MODEL_ZIP = os.path.join(MODEL_DIR)
MODEL_FILE = "model.safetensors"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

def download_folder_from_drive(folder_id, model_dir):
    # Skip download if folder already exists and is not empty
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"Folder '{model_dir}' already exists and is not empty. Skipping download.")
        return

    print("Downloading folder from Google Drive...")
    try:
        subprocess.run([
            "gdown",
            "--folder",
            f"https://drive.google.com/drive/folders/{folder_id}",
            "-O",
            model_dir
        ], check=True)
        print("Download completed successfully.")
        
    except subprocess.CalledProcessError as e:
        print("Download failed:", e)

# Example usage:
FOLDER_ID = "1-2qbQ_ckssma4qXuKne_oT7-rL2vDk_2"
download_folder_from_drive(FOLDER_ID, MODEL_DIR)




# Function to download model from Google Drive
def download_model_from_drive(file_id):
    if not os.path.exists(MODEL_DIR) or not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        os.makedirs(MODEL_DIR, exist_ok=True)  # Automatically create folder if it doesn't exist
        subprocess.run([
            "gdown",
            f"https://drive.google.com/uc?id={file_id}",
            "-O", MODEL_PATH
            ])
        print("Download complete.")
    else:
        print("Model already exists, skipping download.")

# Trigger the download before the app starts
download_model_from_drive("1-bWg2ZkGhDQToWbQOSx47zN1wLadMtQm")


# Initialize FastAPI app
app = FastAPI(title="Text Summarization System", description="Summarize Texts with T5!", version="1.0")

# Load model and tokenizer
loaded_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
loaded_tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

# Ensure the model is on the correct device
device = "cuda" if loaded_model.device.type == "cuda" else "cpu"
model = loaded_model.to(device)

# Mount templates
templates = Jinja2Templates(directory=".")


# Input schema for requests
class DialogueInput(BaseModel):
    dialogue: str


# Clean text function
def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
    text = text.strip().lower()  # Strip and convert to lower case
    return text


# Summarization function
def Summarize_Text(Input_text: str) -> str:
    Input_text = clean_text(Input_text)
    inputs = loaded_tokenizer(Input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate summary
    outputs = loaded_model.generate(inputs["input_ids"], max_length=200,  num_beams=4, early_stopping=True)
    
    summary = loaded_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return summary


# API endpoint for text summarization
@app.post("/summarize/")
async def summarize(Input_text: DialogueInput):
    summary = Summarize_Text(Input_text.dialogue)
    return {"summary": summary}


# HTML UI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("web_app.html", {"request": request})