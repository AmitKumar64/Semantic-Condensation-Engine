# Semantic Condensation Engine

This is a Streamlit web application that utilizes the BART (Bidirectional and Auto-Regressive Transformers) model for text summarization. Users can input text into the application, and it will generate a condensed summary of the input text.

## Setup Instructions

1. **Environment Setup**
   - Ensure you have Python installed on your system.
   - Install the required libraries by running:
     ```
     pip install streamlit transformers
     ```

2. **Download Model Weights**
   - The code utilizes the "facebook/bart-large-cnn" model for summarization. These weights will be downloaded automatically when running the application for the first time.

3. **Running the Application**
   - Run the application script (`app.py`) using the following command:
     ```
     streamlit run app.py
     ```
   - This will start a local web server hosting the application.

4. **Usage**
   - Once the application is running, you will see a text area where you can input your text.
   - Adjust the slider to select the desired length for the summary.
   - Click the "Summarize" button to generate the summary.

## Code Structure

- **app.py**: Contains the Streamlit application code.
  - Imports the necessary libraries and defines a function for text summarization using the BART model.
  - Creates the Streamlit UI elements such as text input, slider for summary length selection, and a button to trigger summarization.
  - Displays warnings for empty input or inputs with fewer than 20 words.
  - Summarizes the input text using the `bart_summarize` function and displays the result.

## Functionality

- **Text Summarization**:
  - Utilizes the BART model to generate a summary of the input text.
  - Allows users to customize the length of the summary.
  - Provides feedback messages for input validation.

## Note

- This application is intended for educational and demonstrational purposes.
- Depending on the length and complexity of the input text, the summarization process may take some time.

## Future Improvements

- Implementation of a more robust error handling mechanism.
- Integration of additional models for text summarization to provide users with more options.
- Enhancements to the user interface for a better user experience.

## Credits

- This application was developed by [Your Name].
- It utilizes the Streamlit library for building interactive web applications and the Hugging Face Transformers library for natural language processing tasks.

## License

[Include license information if applicable.]

## Contact

For any inquiries or feedback, please contact [Your Email Address].



