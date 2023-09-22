import gradio as gr
from fastai.vision.all import *
import numpy as np
import gradio as gr
import json
from os.path import dirname, realpath, join
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
learn = load_learner('./1_resnet34.pkl')
labels = [
    'This is Lebmuernang underripe stage, it has high resistance starch and probiotics',
    'This is Lebmuernang ripe stage, it has high fiber and low sugar',
    'This is Lebmuernang very ripe stage, it has high fiber and antioxidants',
    'This is Lebmuernang over ripe state, it has highest sugar ,lowest fiber and vitamin content',
    'This is PisangAwake underripe stage, it has high resistance starch and probiotics',
    'This is PisangAwake ripe stage, it has high fiber and low sugar',
    'This is PisangAwake very ripe stage, it has high fiber and antioxidants',
    'This is PisangAwake over ripe state, it has highest sugar ,lowest fiber and vitamin content']

def predict(img): 
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
gr_interface = gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(224, 224)),outputs=gr.outputs.Label(num_top_classes=3), title="BananaRipenessClassification",description="* The least ripe side of the banana should be taken.", interpretation="default",examples=[
        ["./1686827532687.jpg"], 
        ["./IMG_20230614_182338.jpg"],
        ["./1686932045978.jpg"],

    ])
gr_interface.launch()