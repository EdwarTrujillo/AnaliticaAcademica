
from pathlib import Path

import pandas as pd
import joblib

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Predicción de aprobación estudiantil")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

modelo = joblib.load(BASE_DIR / "modelo_mlp_estudiantes.joblib")

COLUMNAS = [
    "Tareas",
    "Examenes",
    "Asistencia",
    "Estudio_Horas",
    "Sueno_Horas",
    "Extracurriculares",
]


@app.get("/", response_class=HTMLResponse)
def inicio(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "resultado": None,
            "probabilidad": None,
            "datos": {},
            "error": None,
        },
    )


@app.post("/predecir", response_class=HTMLResponse)
def predecir(
    request: Request,
    Tareas: float = Form(...),
    Examenes: float = Form(...),
    Asistencia: float = Form(...),
    Estudio_Horas: float = Form(...),
    Sueno_Horas: float = Form(...),
    Extracurriculares: float = Form(...),
):
    datos_formulario = {
        "Tareas": Tareas,
        "Examenes": Examenes,
        "Asistencia": Asistencia,
        "Estudio_Horas": Estudio_Horas,
        "Sueno_Horas": Sueno_Horas,
        "Extracurriculares": Extracurriculares,
    }

    datos = pd.DataFrame([datos_formulario], columns=COLUMNAS)

    try:
        prediccion = int(modelo.predict(datos)[0])
        resultado = "Aprobado" if prediccion == 1 else "Reprobado"

        probabilidades = modelo.predict_proba(datos)[0]
        probabilidad = round(
            float(probabilidades[prediccion]) * 100, 2
        )

        error = None

    except Exception as e:
        resultado = None
        probabilidad = None
        error = f"No fue posible generar la predicción: {e}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "resultado": resultado,
            "probabilidad": probabilidad,
            "datos": datos_formulario,
            "error": error,
        },
    )
