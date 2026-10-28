# Sistema de predicción de aprobación estudiantil

Proyecto académico de Inteligencia Artificial que utiliza una red
neuronal MLP para predecir la aprobación o reprobación de estudiantes.

## Tecnologías
- Python
- Scikit-learn
- FastAPI
- HTML
- Render

## Ejecución local

Instalar las dependencias:

pip install -r requirements.txt

Ejecutar la aplicación:

uvicorn main:app --reload

Abrir en el navegador:

http://127.0.0.1:8000

## Modelo
El modelo entrenado se encuentra en:
modelo_mlp_estudiantes.joblib

La aplicación recibe seis variables de entrada y devuelve
la clase predicha por el modelo.