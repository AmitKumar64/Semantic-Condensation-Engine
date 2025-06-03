import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

@st.cache_resource
def load_model():
    """Load and cache the BART model and tokenizer"""
    model_name = "facebook/bart-large-cnn"
    try:
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None, None

def bart_summarize(input_text, max_summary_length):
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return None

    try:
        inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)
        
        # Convert word count to approximate token count (1 word ≈ 1.3 tokens for English)
        max_tokens = int(max_summary_length * 1.3)
        min_tokens = max(1, int(max_summary_length * 0.5))  # Minimum 50% of desired length
        
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_new_tokens=max_tokens,  # Use max_new_tokens instead of max_length
            min_length=min_tokens, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Post-process to get closer to desired word count
        summary_words = summary.split()
        if len(summary_words) > max_summary_length:
            # Truncate to last complete sentence within word limit
            truncated = ' '.join(summary_words[:max_summary_length])
            # Find last sentence ending
            last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
            if last_period > len(truncated) * 0.7:  # If sentence ending is in last 30%
                summary = truncated[:last_period + 1]
            else:
                summary = truncated + '...'
        
        return summary
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            st.error("Text too long for available memory. Try shorter text or reduce summary length.")
        else:
            st.error(f"Generation error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during summarization: {e}")
        return None

# Streamlit App
st.title("🔥 Semantic Condensation Engine")
st.write("Transform lengthy text into concise, meaningful summaries using AI.")

# Initialize session state
if 'example_text' not in st.session_state:
    st.session_state.example_text = ""

# Example text
example_text = """Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect."""

# Add example button
if st.button("📝 Use Example Text"):
    st.session_state.example_text = example_text
    st.rerun()

input_text = st.text_area(
    "Input Text:", 
    value=st.session_state.example_text,
    placeholder="Enter your text here...",
    help="Enter the text you want to summarize (minimum 20 words recommended)"
)

# Length control
default_summary_length = 150
max_summary_length = st.slider(
    "Select Summary Length (in words):", 
    min_value=30, 
    max_value=300, 
    value=default_summary_length,
    help="Choose the approximate number of words in your summary"
)

if st.button("✨ Summarize", type="primary"):
    # Input validation
    if not input_text.strip():
        st.warning("⚠️ Please enter some text for summarization.")
    else:
        with st.spinner("🤖 Summarizing... Please wait."):
            word_count = len(input_text.split())
            
            if word_count > 20:
                summarized_text = bart_summarize(input_text, max_summary_length)
                
                if summarized_text:
                    st.success("✅ Summary completed!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Words", word_count)
                    with col2:
                        summary_word_count = len(summarized_text.split())
                        st.metric("Summary Words", summary_word_count)
                    
                    st.subheader("📄 Summarized Text:")
                    st.write(summarized_text)
                    
                    # Compression ratio
                    compression_ratio = round((1 - summary_word_count/word_count) * 100, 1)
                    st.info(f"📊 Compression: {compression_ratio}% reduction in length")
            else:
                st.warning("⚠️ Text too short for meaningful summarization. Please enter at least 20 words.")

# Footer
st.markdown("---")
st.markdown("*Powered by Facebook's BART-Large-CNN model*")

st.markdown(
    """
    <div style='text-align: center; margin-top: 30px; 
                padding: 20px; font-size: 18px; color: #d1d1d1;'>
        <em>Made by Amit, Raj, Shivam and Mayank</em>
    </div>
    """, 
    unsafe_allow_html=True
)