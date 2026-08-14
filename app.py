from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import joblib
import pandas as pd
import sqlite3
import warnings

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI()

# Mount static directory for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Global model evaluation benchmark
MODEL_ACCURACY = 96.38

def init_results_database():
    """Ensures the SQLite database and table exist, including the confidence score column."""
    conn = sqlite3.connect("diabetes_records.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            pregnancies REAL,
            glucose REAL,
            blood_pressure REAL,
            skin_thickness REAL,
            insulin REAL,
            bmi REAL,
            age REAL,
            job_type INTEGER,
            prediction_result TEXT,
            confidence_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        cursor.execute("ALTER TABLE patient_predictions ADD COLUMN confidence_score REAL")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

# Initialize database on startup
init_results_database()

def log_prediction_to_db(patient_name, vitals, result_str, confidence):
    """Stores patient vitals, prediction results, and confidence score into SQLite."""
    conn = sqlite3.connect("diabetes_records.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patient_predictions 
        (patient_name, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, age, job_type, prediction_result, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_name,
        vitals['Number of Pregnancies'],
        vitals['Glucose Level'],
        vitals['Blood Pressure'],
        vitals['Skin Thickness'],
        vitals['Insulin'],
        vitals['BMI'],
        vitals['Age'],
        vitals['Job Type'],
        result_str,
        confidence
    ))
    conn.commit()
    conn.close()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "model_accuracy": MODEL_ACCURACY}
    )

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    conn = sqlite3.connect("diabetes_records.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_predictions ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()
    
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"request": request, "records": records, "model_accuracy": MODEL_ACCURACY}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    patient_name: str = Form(...),
    pregnancies: int = Form(...),
    glucose: float = Form(...),
    bp: float = Form(...),
    skin: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    age: int = Form(...),
    job_type: int = Form(...)
):
    vitals = {
        'Number of Pregnancies': pregnancies,
        'Glucose Level': glucose,
        'Blood Pressure': bp,
        'Skin Thickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'Age': age,
        'Job Type': job_type
    }
    
    df_input = pd.DataFrame([vitals])
    df_scaled = scaler.transform(df_input)
    
    prediction = model.predict(df_scaled)
    probs = model.predict_proba(df_scaled)[0]
    confidence = float(max(probs) * 100)
    
    if prediction[0] == 1:
        result_text = f"⚠️ High Risk for {patient_name}: The AI model predicts Diabetes (Confidence: {confidence:.2f}%)."
        result_str = "Diabetic"
        is_diabetic = True
    else:
        result_text = f"✅ Low Risk for {patient_name}: The AI model predicts Non-Diabetic (Confidence: {confidence:.2f}%)."
        result_str = "Non-Diabetic"
        is_diabetic = False
        
    log_prediction_to_db(patient_name, vitals, result_str, confidence)
        
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request, 
            "prediction_text": result_text,
            "is_diabetic": is_diabetic,
            "model_accuracy": MODEL_ACCURACY
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)