from pathlib import Path
import json, math
import pandas as pd
import joblib
import gradio as gr

BASE = Path(__file__).resolve().parent
ARTEFACTOS = BASE / 'artefactos'
SCHEMA = json.loads((ARTEFACTOS/'esquema.json').read_text(encoding='utf-8'))
METRICS = json.loads((ARTEFACTOS/'metricas.json').read_text(encoding='utf-8'))
# Cargar únicamente artefactos propios y confiables; joblib no es seguro con archivos desconocidos.
PIPELINE = joblib.load(ARTEFACTOS/'pipeline.joblib')

def predict_order(unidades, carga_bodega_pct, stock_disponible, prioridad):
    values = dict(unidades=unidades,carga_bodega_pct=carga_bodega_pct,stock_disponible=stock_disponible)
    for k,v in values.items():
        bounds = SCHEMA['numericas'][k]
        if v is None or not math.isfinite(v) or v != int(v) or not bounds['min'] <= v <= bounds['max']:
            raise ValueError(f'{k}: ingresar un entero entre {bounds["min"]} y {bounds["max"]}.')
        values[k] = int(v)
    if prioridad not in SCHEMA['prioridades']:
        raise ValueError('Prioridad no válida.')
    values['prioridad'] = prioridad
    row = pd.DataFrame([values],columns=SCHEMA['variables'])
    p = float(PIPELINE.predict_proba(row)[0,1])
    return p, p >= SCHEMA['umbral']

def predict_ui(unidades,carga_bodega_pct,stock_disponible,prioridad):
    try:
        p,flag = predict_order(unidades,carga_bodega_pct,stock_disponible,prioridad)
    except (ValueError,TypeError) as exc:
        raise gr.Error(str(exc)) from exc
    status = 'SÍ — alerta de riesgo de error' if flag else 'NO — sin alerta de riesgo'
    advice = 'Se sugiere una revisión adicional del pedido.' if flag else 'Mantener los controles habituales; no garantiza ausencia de error.'
    return status, round(100*p,2), round(100*(1-p),2), advice

with gr.Blocks(title='LogistiX AI | Packing') as demo:
    gr.Markdown('# LogistiX AI\n**G4 · Prototipo 1 | Predicción de errores de packing**\n\nEvaluación académica · Ingrese los datos del pedido antes de iniciar la preparación.')
    with gr.Row():
        with gr.Column():
            inputs = []
            for key,label in [('unidades','Unidades del pedido'),('carga_bodega_pct','Carga de bodega (%)'),('stock_disponible','Stock disponible')]:
                b = SCHEMA['numericas'][key]
                inputs.append(gr.Number(label=label, value=b['mediana'], minimum=b['min'], maximum=b['max'], precision=0, info=f'Rango observado: {b["min"]}–{b["max"]}'))
            inputs.append(gr.Dropdown(choices=SCHEMA['prioridades'],value='Normal',label='Prioridad'))
            btn = gr.Button('Predecir riesgo',variant='primary')
        with gr.Column():
            result = gr.Textbox(label='Resultado',interactive=False)
            prob = gr.Number(label='Probabilidad estimada de error (%)',interactive=False)
            no_prob = gr.Number(label='Probabilidad estimada sin error (%)',interactive=False)
            advice = gr.Textbox(label='Acción sugerida',interactive=False)
    gr.Markdown(f'**Umbral de alerta:** {SCHEMA["umbral"]:.0%} (seleccionado con F2 en validación). Un umbral menor al 50% permite alertar más pedidos, pero genera muchos falsos positivos. **Uso experimental:** las probabilidades no se han calibrado; el ROC-AUC de prueba es {METRICS["prueba_umbral"]["roc_auc"]:.3f}. No utilizar como criterio único de operación.')
    btn.click(predict_ui,inputs=inputs,outputs=[result,prob,no_prob,advice])
    gr.Examples(examples=[[23,94,117,'Alta'],[37,71,219,'Urgente'],[113,47,158,'Urgente']],inputs=inputs)
    gr.Markdown(f'Versión del modelo: {SCHEMA["version"]} · XGBoost + preprocesamiento guardado en pipeline.')

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1',share=False)
