import streamlit as sl
from dataloader import *
from model_setup import *
from train import *

sl.markdown("""
    <style>
        .title-header {
            text-align: center;
            color: #1f77b4;
            font-size: 40px;
            font-weight: 700;
        }
        .subtext {
            text-align: center;
            color: #555555;
        }
        .stButton>button {
            display: block;
            margin: auto;
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

sl.markdown('<div class="title-header">Pneumonia DETECTOR</div>', unsafe_allow_html=True)
sl.markdown('<div class="subtext">By: Sumukh Sudhir Jagirdar, Aedin Cowan, Brain Chan, Joys James</div>', unsafe_allow_html=True)

sl.text_area("IMAGE option coming soon..")

def detect_image():
    sl.text("HELLO")
sl.button("Detect", on_click=detect_image)
