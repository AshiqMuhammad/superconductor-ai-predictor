import gradio as gr
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("superconductor_model.pkl")
features = joblib.load("top_features.pkl")

# =========================
# INTERPRETATION
# =========================
def interpret_tc(tc):
    if tc < 10:
        return "Low-Temperature Superconductor"
    elif tc < 30:
        return "Medium-Temperature Superconductor"
    elif tc < 77:
        return "High-Temperature Superconductor"
    else:
        return "Ultra High-Temperature Superconductor"

# =========================
# SUPERCONDUCTOR CHANCE (NEW IMPORTANT FEATURE)
# =========================
def superconductivity_chance(tc):
    # normalized probability (0–100%)
    prob = min(max((tc / 100) * 100, 0), 100)
    return round(prob, 2)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_tc(*inputs):
    inputs = np.array(inputs).reshape(1, -1)
    tc = float(model.predict(inputs)[0])

    label = interpret_tc(tc)
    chance = superconductivity_chance(tc)

    return f"""
🧊 SUPERCONDUCTOR AI RESULT

────────────────────────────
Critical Temperature: {tc:.2f} K
Material Type: {label}
Superconductivity Chance: {chance} %

────────────────────────────
AI Model: Random Forest Regressor
Physics ML System
"""

# =========================
# CLEAN INPUT UI
# =========================
inputs = [gr.Number(label=f.replace("_"," ").title()) for f in features]

# =========================
# PROFESSIONAL UI (NO SHARE BUTTON, CLEAN OUTPUT)
# =========================
app = gr.Interface(
    fn=predict_tc,
    inputs=inputs,
    outputs=gr.Textbox(lines=8),
    title="🧊 Superconductor AI System",
    description="Industrial-level physics AI model for predicting superconducting materials",
    theme=gr.themes.Soft()
)

app.launch()