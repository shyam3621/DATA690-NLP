import streamlit as st
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.translate.bleu_score import sentence_bleu
import fitz  # PyMuPDF
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import time
from gtts import gTTS
from io import BytesIO
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
from googletrans import Translator
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')




def preprocess_and_analyze_text(input_text):
    # Tokenization
    words = word_tokenize(input_text)
    sentences = sent_tokenize(input_text)

    # Removing stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.lower() not in punctuation]

    # Calculating word frequency
    word_freq = Counter(filtered_words)

    # Calculating sentence count and average sentence length
    num_sentences = len(sentences)
    total_words = len(filtered_words)
    avg_sentence_length = total_words / num_sentences if num_sentences > 0 else 0

    # Finding most common words
    most_common_words = word_freq.most_common(5)

    # Checking for meaningful content
    meaningful_content = True if total_words > 0 and num_sentences > 0 else False

    # Constructing analysis results
    analysis_results = {
        'Total Sentences': num_sentences,
        'Total Words': total_words,
        'Average Sentence Length': avg_sentence_length,
        'Most Common Words': most_common_words,
        'Meaningful Content': meaningful_content
    }

    return analysis_results




def fetch_text_from_url(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text content from specific HTML tags (you can customize this based on your requirements)
            text_content = []
            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                text_content.append(tag.get_text().strip())

            # Combine the text content into a single string
            text = '\n'.join(text_content)

            # Return the extracted text
            return text
        else:
            # If the request was not successful, print an error message
            st.error(f"Failed to fetch content from URL. Status code: {response.status_code}")
            return None
    except Exception as e:
        # Handle any exceptions that may occur during the process
        st.error(f"Error fetching content from URL: {e}")
        return None


# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pdf_text = ""

    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        pdf_text += page.get_text()

    return pdf_text


# Function for Hugging Face Abstractive Summarization using BART
def summarize_with_bart(original_text):
    try:
        start_time = time.time()
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        inputs = tokenizer.encode(original_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=500, min_length=500, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        end_time = time.time()
        execution_time = end_time - start_time
        return summary, execution_time
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None, None



# Function to convert Text to Speech {Additional Feature}
def text_to_speech(text):
    tts = gTTS(text, lang='en')
    audio_bytes_io = BytesIO()
    tts.write_to_fp(audio_bytes_io)
    return audio_bytes_io.getvalue()


# Function for Animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://lottie.host/90ff5ee7-1753-4672-9990-3d3b45d7e4e2/0H1gQB5t3v.json")

#https://lottie.host/def31937-424f-48ce-8fda-cb90db6cacf0/9ov1bj4lt5.json


translator = Translator()


def styled_write(text, font_size=18, text_color="white", background_color="black"):
    styled_text = (
        f"<div style='font-size: {font_size}px; color: {text_color}; "
        f"background-color: {background_color}; padding: 10px;'>{text}</div>"
    )
    st.write(styled_text, unsafe_allow_html=True)


# Streamlit app starts from here
def main():

    with st.sidebar:
        selected = option_menu(
            menu_title = "Menu",
            options = ["Home","Summarize", "Speech", "Translate"],
            icons = ["house","book","mic","translate"],
            default_index = 0,
            )

    # HOME PAGE CODE for Streamlit app
    if selected == "Home":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title(" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; SNAP SUMM")

        st_lottie(lottie_coding, height = 425, key= "coding")

        sentences = [
            "Distill the essence of information,",
            "Navigate the sea of words effortlessly,",
            "Embrace the clarity of concise insights."
        ]

        animation_placeholder = st.empty()
        index = 0

        # Using infinite loop for continuous animation
        while True:
            sentence = sentences[index]

            styled_sentence = f"<p style='font-family: cursive; font-size: 25px; text-align: center;'>{sentence}</p>"
            animation_placeholder.markdown(styled_sentence, unsafe_allow_html=True)

            index = (index + 1) % len(sentences)
            time.sleep(2)

        st.text("")
        st.text("")
        st.text("")


    # Summarize Page code for Streamlit app
    if selected == "Summarize":
        background_image_url = ""
        background_style = f"""
            <style>
                .stApp {{
                    background-image: url('{background_image_url}');
                    background-size: cover;
                }}
            </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)

        st.markdown(
            """
            <style>
                h1 {{
                    color: #FFFFFF; /* White */
                    text-align: center;
                    font-family: 'Roboto', sans-serif;  /* Replace 'Your Font Here' with the desired font */
                    font-size: 50px;  /* Adjust the font size as needed */
                }}
                .quote {{
                    font-style: italic;
                    text-align: center;
                    color: #FFFFFF; /* White */
                    margin-top: 20px;
                    margin-bottom: 30px;
                    font-family: 'cursive';  /* Set to a cursive font */
                    font-size: 20px;  /* Adjust the font size as needed */
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("SNAP SUMM")


        st.text("")
        st.text("")

        # Input choice for Text or PDF file
        input_choice = st.radio("Choose your input format:", ("Text", "PDF File","URL"))
        st.markdown("<style>label {font-size: 50px;}</style>", unsafe_allow_html=True)

        original_text=""
        # Input text
        if input_choice == "Text":
            original_text = st.text_area("Enter the original text:")
        elif input_choice == "PDF File":
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if uploaded_file:
                original_text = extract_text_from_pdf(uploaded_file)
        elif input_choice == "URL":
            url = st.text_input("Enter the URL of the webpage:")
            if st.button("Fetch Text"):
                if url:
                    # Display loading spinner while fetching text
                    with st.spinner("Fetching text from URL..."):
                        # Fetch text from the URL
                        fetched_text = fetch_text_from_url(url)
                        if fetched_text:
                            original_text = fetched_text
                            st.success("Text fetched from URL.")
                else:
                    st.warning("Please enter a valid URL.")


        if original_text:
            analysis_results = preprocess_and_analyze_text(original_text)

            # Display analysis results
            st.header("Analysis Results")
            st.write("Total Sentences:", analysis_results['Total Sentences'])
            st.write("Total Words:", analysis_results['Total Words'])
            st.write("Average Sentence Length:", analysis_results['Average Sentence Length'])
            st.write("Most Common Words:", analysis_results['Most Common Words'])
            st.write("Meaningful Content:", analysis_results['Meaningful Content'])
            st.write(original_text)

        # # Reference text (optional) This is used to Calculate the metrics later in the code if entered by the user
        # reference_summary = st.text_area("Enter the reference summary (ground truth):")


        # Execution of Summaries
        if st.button("Generate Summaries"):
            # Display loading spinner while processing
            with st.spinner("Generating Summaries..."):
                
                # Hugging Face - BART
                bart_summary, bart_time = summarize_with_bart(original_text)
                st.header("Hugging Face - BART Summarization")
                st.markdown(
                f'<div style="background-color: #333333; color: #ffffff; padding: 10px; border-radius: 5px;">{bart_summary}</div>',
                unsafe_allow_html=True
                )
               
               
                # If reference summary is provided by user, calculate BLEU metrics
                

        centered_button_style = """
            <style>
                div.stButton > button {
                    display: block;
                    margin: 0 auto;
                }
            </style>
        """
        st.markdown(centered_button_style, unsafe_allow_html=True)

    # Speech code for Streamlit
    elif selected == "Speech":
       # Set background image from URL
        background_image_url = ""
        background_style = f"""
            <style>
                .stApp {{
                    background-image: url('{background_image_url}');
                    background-size: cover;
                }}
            </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)

        # Header styling
        st.markdown(
            """
            <style>
                h1 {{
                    color: #FFFFFF; /* White */
                    text-align: center;
                    font-family: 'Roboto', sans-serif;  /* Replace 'Your Font Here' with the desired font */
                    font-size: 50px;  /* Adjust the font size as needed */
                }}
                .quote {{
                    font-style: italic;
                    text-align: center;
                    color: #FFFFFF; /* White */
                    margin-top: 20px;
                    margin-bottom: 30px;
                    font-family: 'cursive';  /* Set to a cursive font */
                    font-size: 20px;  /* Adjust the font size as needed */
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        
        col1, col2, col3 = st.columns([1, 2,3])

        with col2:
            st.title("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Speech&nbsp;&nbsp;&nbsp;")

        input_text = st.text_area("Enter the text")
        centered_button_style = """
            <style>
                div.stButton > button {
                    display: block;
                    margin: 0 auto;
                }
            </style>
        """
        st.markdown(centered_button_style, unsafe_allow_html=True)

        if st.button("Speech"):
            audio_data = text_to_speech(input_text)
            st.audio(audio_data, format="audio/mp3")


    # Translate code for Streamlit 
    elif selected == "Translate":

        background_image_url = ""
        background_style = f"""
            <style>
                .stApp {{
                    background-image: url('{background_image_url}');
                    background-size: cover;
                }}
            </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)

        st.markdown(
            """
            <style>
                h1 {{
                    color: #FFFFFF; /* White */
                    text-align: center;
                    font-family: 'Roboto', sans-serif;  /* Replace 'Your Font Here' with the desired font */
                    font-size: 50px;  /* Adjust the font size as needed */
                }}
                .quote {{
                    font-style: italic;
                    text-align: center;
                    color: #FFFFFF; /* White */
                    margin-top: 20px;
                    margin-bottom: 30px;
                    font-family: 'cursive';  /* Set to a cursive font */
                    font-size: 20px;  /* Adjust the font size as needed */
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        
        col1, col2, col3 = st.columns([1, 2,3])

        with col2:
            st.title("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Translate&nbsp;&nbsp;&nbsp;")


        input_text_translate = st.text_area("Enter the text for translation")

        language_options = {
            "Select Language": "",
            "French": "fr",
            "Spanish": "es",
            "German": "de",
            "Italian": "it",
            "Russian": "ru",
            "Telugu": "te",
        }

        target_language = st.selectbox("Select the target language", list(language_options.keys()))

        if st.button("Translate"):
            if target_language == "Select Language":
                st.warning("Please select a target language.")
            else:
                target_language_code = language_options[target_language]

                # Translating the input text
                translated_text = translator.translate(input_text_translate, dest=target_language_code).text
                st.header("Translated Text")
                st.markdown(
                    f'<div style="background-color: #333333; color: #ffffff; padding: 10px; border-radius: 5px;">{translated_text}</div>',
                    unsafe_allow_html=True
                )


# Calling the main function
if __name__ == "__main__":
    main()
