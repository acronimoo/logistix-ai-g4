# LogistiX AI — G4 / Prototipo 1

Evaluación N.º 4 de IA: clasificación binaria de errores de packing y probabilidad estimada. El predictor de atraso no forma parte del alcance.

## 1. Entrenamiento en Google Colab (requisito docente)
1. Abrir `LogistiX_Entrenamiento.ipynb` en Google Colab.
2. Ejecutar las celdas en orden y cargar `LogistiX.xlsx` cuando se solicite.
3. Descargar `artefactos.zip` desde la última celda y reemplazar los archivos de la carpeta local `artefactos` por los recién generados. Mantener los tres archivos juntos.
4. Revisar métricas, matriz de confusión y explicación del umbral antes de la exposición.

El notebook se entrega preparado para Colab, pero no se ha ejecutado en ese servicio; el entrenamiento equivalente se ejecutó localmente en el entorno de creación. Se incluyen los artefactos obtenidos de ese entrenamiento.

## 2. Ejecutar la aplicación localmente (Windows PowerShell)
Desde la carpeta extraída, con Python 3.11–3.13 de 64 bits (entorno de prueba: 3.13.5):

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```
Abrir la dirección que aparece en la consola (habitualmente http://127.0.0.1:7860). Para detener: Ctrl+C.

Si prefieres entrenar localmente: `.\.venv\Scripts\python.exe entrenar.py`.
Para pruebas de integración: `.\.venv\Scripts\python.exe -m unittest test_app -v`.

## 3. Decisiones y evidencia
- Objetivo: `error_preparacion` (0 = no, 1 = sí).
- Predictoras: unidades, carga de bodega, stock disponible y prioridad. Se define la predicción antes de iniciar la preparación. Se elimina `pedido_id` (identificador), `distancia_km` (más asociada al transporte y sin justificación causal directa para packing), `tiempo_picking_min` (duración observada que aún no se conoce al inicio) y `entrega_atrasada` (resultado posterior y objetivo de otro prototipo). Las decisiones son de diseño; no afirman que las variables eliminadas no tengan asociación estadística.
- El dataset tiene 1.200 registros, 0 nulos, 0 filas duplicadas, 242 errores y 958 sin errores. Partición estratificada 60/20/20 (720/240/240). Prioridad se transforma con OneHotEncoder. Pipeline conserva preprocesamiento y XGBoost juntos; evita inconsistencias y ajuste del preprocesamiento con datos de test.
- Umbral de alerta 0,10, elegido por F2 en validación entre 0,10 y 0,80 con incrementos de 0,05. F2 enfatiza recall. En prueba: ROC-AUC 0,524; recall 87,50%; precisión 20,49%; exactitud 29,58%; 42 VP, 163 FP, 6 FN y 29 VN. Exactitud con umbral 0,50: 80,42%, recall 4,17%. Baseline sin errores: 80% de exactitud y 0% recall en test. La capacidad discriminativa es baja; no recomendar el uso productivo. El umbral es experimental, no un umbral óptimo de negocio, y los porcentajes no están calibrados.
- Dos limitaciones para el informe: (1) baja capacidad discriminativa y alto costo potencial de falsas alertas, (2) tamaño/cobertura limitados del dataset, sin fecha ni variables operacionales como experiencia del preparador, complejidad SKU y turno, que impiden comprobar generalización temporal y a otras bodegas.

## 4. MLOps
Implementado: código reproducible, semilla, hash SHA-256 del dataset, registro de versiones, métricas y esquema de entrada, persistencia del pipeline, validación y pruebas unitarias. `artefactos/metricas.json` contiene resultados reproducibles del modelo. Propuesto para producción: Git y CI, revisión de nuevas versiones antes del despliegue, control de acceso, captura de resultados reales, monitoreo de drift y de recall/precisión por período, reentrenamiento evaluado y rollback a una versión aprobada. No se afirma que estas prácticas propuestas ya estén implementadas.

## 5. Entregables pendientes de completar por el equipo
Informe técnico (formato vertical oficial INACAP 60 años, extensión sugerida 6–10 páginas sin portada, conclusiones y referencias) y presentación de máximo 7 diapositivas. Usar resultados reales de la corrida de Colab. No se incluyen aquí la plantilla institucional ni la presentación.

Fuentes: Evaluación N.º 4 de IA y LogistiX.xlsx entregados por el docente. Para fundamentar las herramientas, consultar documentación oficial: https://xgboost.readthedocs.io/ ; https://scikit-learn.org/stable/ ; https://www.gradio.app/docs ; https://colab.research.google.com/ .
