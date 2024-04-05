import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

def bart_summarize(input_text, max_summary_length):
    model_name = "facebook/bart-large-cnn"
    try:
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

    inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_summary_length, min_length=1, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit App
st.title("Semantic Condensation Engine")

input_text = st.text_area("Input Text:", "Enter your text here...")

# Add length tool
default_summary_length = 150  # Set a default value
max_summary_length = st.slider("Select Summary Length:", min_value=50, max_value=500, value=default_summary_length)

if st.button("Summarize"):
    # Input validation
    if not input_text.strip():
        st.warning("Please enter some text for summarization.")
    else:
        with st.spinner("Summarizing... Please wait."):
            word_count = len(input_text.split())

            if word_count > 20:
                summarized_text = bart_summarize(input_text, max_summary_length)

                if summarized_text:
                    st.subheader("Summarized Text:")
                    st.write(summarized_text)
            else:
                st.warning("The given input cannot be summarized.")
