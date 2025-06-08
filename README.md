# 🧠 Visual Question Answering Web App

This is a simple web application built using **FastAPI** that allows users to:
- Upload an image
- Ask multiple questions about it
- Get visual question-answering (VQA) responses using the `dandelin/vilt-b32-finetuned-vqa` model
- See image preview and all previous Q&A pairs in a clean UI

---

## 🚀 Features

✅ Upload a single image  
✅ Ask multiple questions about the image  
✅ View image preview  
✅ See full Q&A history  
✅ Upload a new image to reset session  

---

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Frontend**: HTML + CSS (Jinja2 Templates)
- **Model**: [ViLT VQA - HuggingFace](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- **Serving**: Uvicorn

---

## 📦 Installation

```bash
# 1. Clone this repo
git clone https://github.com/yourusername/vqa-app.git
cd vqa-app

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the App
uvicorn main:app --reload
```
