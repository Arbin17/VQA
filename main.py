from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch
import io
import os
import uuid

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global session-like storage
session_data = {
    "image_bytes": None,
    "image_path": None,
    "qa_history": []
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "qa_history": [],
        "image_uploaded": False,
        "image_url": None
    })

@app.post("/upload-image/", response_class=HTMLResponse)
async def upload_image(request: Request, image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_id = str(uuid.uuid4()) + ".jpg"
    image_path = os.path.join(UPLOAD_DIR, image_id)

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    session_data["image_bytes"] = image_bytes
    session_data["image_path"] = image_path
    session_data["qa_history"] = []

    return RedirectResponse(url="/qa", status_code=303)

@app.get("/qa", response_class=HTMLResponse)
async def ask_page(request: Request):
    if session_data["image_bytes"] is None:
        return RedirectResponse(url="/", status_code=303)

    image_url = "/" + session_data["image_path"]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "qa_history": session_data["qa_history"],
        "image_uploaded": True,
        "image_url": image_url
    })

@app.post("/ask-question/", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    if session_data["image_bytes"] is None:
        return RedirectResponse(url="/", status_code=303)

    image_pil = Image.open(io.BytesIO(session_data["image_bytes"])).convert("RGB")
    inputs = processor(image_pil, question, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        answer = model.config.id2label[predicted_idx]

    session_data["qa_history"].append((question, answer))
    return RedirectResponse(url="/qa", status_code=303)
