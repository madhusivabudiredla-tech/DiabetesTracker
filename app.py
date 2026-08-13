from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import joblib
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Initialize the FastAPI app
app = FastAPI()

# Mount the static directory to serve the CSS file
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize HTML templates
templates = Jinja2Templates(directory="templates")

# Load your winning model
model = joblib.load('final_diabetes_prediction_model.pkl')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    pregnancies: int = Form(...),
    glucose: float = Form(...),
    bp: float = Form(...),
    skin: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    age: int = Form(...),
    job_type: int = Form(...)
):
    # Gather data from the HTML form
    input_data = {
        'Number of Pregnancies': [pregnancies],
        'Glucose Level': [glucose],
        'Blood Pressure': [bp],
        'Skin Thickness': [skin],
        'Insulin': [insulin],
        'BMI': [bmi],
        'Age': [age],
        'Job Type': [job_type]
    }
    
    df_input = pd.DataFrame(input_data)
    
    # Make the prediction using the loaded model
    prediction = model.predict(df_input)
    
    # Format the result for the frontend
    if prediction[0] == 1:
        result_text = "⚠️ High Risk: The AI model predicts Diabetes."
        is_diabetic = True
    else:
        result_text = "✅ Low Risk: The AI model predicts Non-Diabetic."
        is_diabetic = False
        
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request, 
            "prediction_text": result_text,
            "is_diabetic": is_diabetic
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)