# deploy_FastAPI.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import dataset

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load the model
def load_model():
    """Load the trained model from a file"""
    with open('trained_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Render the index page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, sms: str = Form(...)):
    """Handle the prediction request"""
    try:
        # Ensure the input is a string
        sms_input = str(sms)

        cleaned_sms = dataset.clean_text(sms_input, apply_stemming=True, apply_lemmatization=False)

        # Make a prediction using the model
        result = model.predict([cleaned_sms])[0]

        # Render the result
        return templates.TemplateResponse("index.html", {"request": request, "result": result})
    except ValueError:
        # Handle invalid input
        return templates.TemplateResponse("index.html", {"request": request, "result": "Invalid input"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")