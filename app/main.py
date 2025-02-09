from fastapi import FastAPI
import uvicorn
from app.model import train_model, load_model, preprocess_text
import gradio as gr
import pandas as pd
import re
import os
import app.config as conf
from pydantic import BaseModel

app = FastAPI()

# Endpoint to predict sentiment
#Abbiamo usato questa funzione per testare il funzionamento del modello in base a un testo che li abbiamo passato
@app.post("/predict/")
def predict(text: str):
    model = load_model()
    prediction = model.predict([text])[0]
    return {"prediction": prediction}

class dataFE_model(BaseModel):
    testo: str
    model:str

# Gradio interface
def predict_sentiment(text, model_choice):
    print(model_choice)
    if model_choice == "low" and os.path.exists(conf.MODEL_LOW_PATH) and os.path.getsize(conf.MODEL_LOW_PATH) == 0:
        train_model("low")
    elif model_choice == "medium" and os.path.exists(conf.MODEL_MEDIUM_PATH) and os.path.getsize(conf.MODEL_MEDIUM_PATH) == 0:
        train_model("medium")
    elif model_choice == "high" and os.path.exists(conf.MODEL_HIGH_PATH) and os.path.getsize(conf.MODEL_HIGH_PATH) == 0:
        train_model("high")
    

    text__cleaned = re.sub(r'\s+', ' ',text)  # Rimuove spazi multipli
    text__cleaned = re.sub(r'[^\w\s]', '', text)  # Rimuove simboli

    model = load_model(model_choice)

    return model.predict([text__cleaned])[0]

if __name__ == "__main__":
    with gr.Blocks() as interface:
        gr.Markdown("# Political Sentiment Analysis")
        gr.Markdown("Predict if a comment or tweet is written by a Republican or Democrat.")
        
        model_choice = gr.Radio(
            choices=["low", "medium", "high"],
            label="Choose a Model",
            type="value"
        )
        
        text_input = gr.Textbox(label="Enter your text here")
        
        submit_button = gr.Button("Submit", variant="primary")
        
        output = gr.Label()
        
        submit_button.click(predict_sentiment, inputs=[text_input, model_choice], outputs=output)

    interface.launch()

    uvicorn.run(app, host="0.0.0.0", port=8000)