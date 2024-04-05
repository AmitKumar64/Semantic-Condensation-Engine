# import streamlit as st
# import sys
# import os
# from src.TextSummarization.pipeline.prediction import PredictionPipeline

# if 'output' not in st.session_state:
#     st.session_state.output = ""

# def predict(text):
#     try:
#         obj = PredictionPipeline()
#         output = obj.predict(text)
#         return output
#     except Exception as e:
#         raise e

# st.write("# Text Summarizer")
# st.write("""
# AI-Based Tool to Summarize Text using NLP techniques
# """)
         
# user_input = st.text_area(label="Enter Text to Summarize :pencil:", value="", height=20)
# button = st.button(label="Summarize")
# if button: 
#     st.session_state.output = "Loading..."
#     output = predict(user_input)
#     st.session_state.output = output

# st.write("Output: ", st.session_state.output)


# # button1 = st.button(label="Counter")
# # if ('count' not in st.session_state) :
# #     st.session_state.count = 0

# # if button1:
# #     st.session_state.count += 1

# # st.write("Count: ", st.session_state.count)


# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

import streamlit as st
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

def bart_summarize(input_text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit App
st.title("Text Summarization with Streamlit")

input_text = st.text_area("Input Text:", "Enter your text here...")

if st.button("Summarize"):
    summarized_text = bart_summarize(input_text)
    st.subheader("Summarized Text:")
    st.write(summarized_text)
