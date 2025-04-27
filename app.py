import openai
import streamlit as st
import speech_recognition as sr
from PIL import Image
from io import BytesIO

# Set your OpenAI API Key
openai.api_key = "your_openai_api_key_here"

# Function for text query processing
def process_text_query(user_input):
    response = openai.Completion.create(
        engine="gpt-4-turbo",
        prompt=user_input,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function for image processing (using GPT-4 Vision)
def process_image(image):
    # Convert image to a format GPT-4 can understand, usually base64 encoding
    image = Image.open(image)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()

    # Send to GPT-4 Vision (image analysis via OpenAI)
    response = openai.Image.create(
        model="gpt-4-vision",
        image=img_bytes
    )
    return response['data'][0]['text']

# Function for speech to text conversion (using Whisper)
def process_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

    try:
        # Using Whisper for transcription
        text = recognizer.recognize_whisper(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "Sorry, I'm having trouble with the speech recognition service."
    
# Streamlit Interface
def main():
    st.title("Multimodal AI Chatbot")
    mode = st.radio("Select Input Mode", ("Text", "Image", "Voice"))

    if mode == "Text":
        user_input = st.text_input("Ask me anything:")
        if user_input:
            response = process_text_query(user_input)
            st.write("Response:", response)

    elif mode == "Image":
        uploaded_file = st.file_uploader("Upload a car image for damage estimation", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            response = process_image(uploaded_file)
            st.write("Response:", response)

    elif mode == "Voice":
        st.button("Start Listening")
        text = process_voice_input()
        if text:
            st.write("You said:", text)
            response = process_text_query(text)
            st.write("Response:", response)

if __name__ == "__main__":
    main()
