import speech_recognition as sr
import pyttsx3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re

class HealthcareChatbot:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.intents_df = self.load_intents('MedDatasetFinal_modeled.csv')  # Replace with your actual filename
        self.symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']

    def load_intents(self, file_path):
        df = pd.read_csv(file_path)
        print("Columns in the CSV file:", df.columns)
        print("First few rows of the dataframe:")
        print(df.head())
        return df

    def speech_to_text(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that.")
                return None
            except sr.RequestError:
                print("Sorry, there was an error with the speech recognition service.")
                return None

    def text_to_speech(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def find_matching_diseases(self, symptoms):
        mask = self.intents_df[self.symptom_columns].apply(lambda x: x.str.contains('|'.join(symptoms), case=False, na=False)).any(axis=1)
        return self.intents_df[mask]

    def generate_response(self, disease):
        symptom_list = ', '.join([symptom for symptom in disease[self.symptom_columns] if pd.notna(symptom) and symptom.lower() != 'unknown'])
        medication_columns = ['Medication_1', 'Medication_2', 'Medication_3', 'Medication_4']
        medication_list = ', '.join([medication for medication in disease[medication_columns] if pd.notna(medication) and medication.lower() != 'unknown'])
        
        response = f"Based on your symptoms, it could be {disease['Disease']}. Common symptoms include {symptom_list}. "
        if medication_list:
            response += f"Typical treatments might involve {medication_list}. "
        response += "Please consult a healthcare professional for proper diagnosis and treatment."
        return response

    def chat(self):
        print("Healthcare Chatbot: Hello! How can I assist you today? Please describe your main symptom.")
        self.text_to_speech("Hello! How can I assist you today? Please describe your main symptom.")
        
        symptoms = []
        while True:
            user_input = self.speech_to_text()
            if user_input is None:
                continue
            
            preprocessed_input = self.preprocess_text(user_input)
            symptoms.append(preprocessed_input)
            
            print("Healthcare Chatbot: Any other symptoms? (Say 'no' if you're finished listing symptoms)")
            self.text_to_speech("Any other symptoms?")
            
            user_input = self.speech_to_text()
            if user_input and user_input.lower() == 'no':
                break
        
        if not symptoms:
            print("Healthcare Chatbot: No symptoms were provided. I can't make a prediction without any symptoms.")
            self.text_to_speech("No symptoms were provided. I can't make a prediction without any symptoms.")
            return

        matching_diseases = self.find_matching_diseases(symptoms)
        
        if matching_diseases.empty:
            response = "I couldn't find any matching conditions for your symptoms. Please consult a healthcare professional for proper diagnosis."
        else:
            # Select the disease that matches the most symptoms
            disease_match_count = matching_diseases[self.symptom_columns].apply(lambda x: x.str.contains('|'.join(symptoms), case=False, na=False).sum(), axis=1)
            best_match = matching_diseases.loc[disease_match_count.idxmax()]
            response = self.generate_response(best_match)
        
        print(f"Healthcare Chatbot: {response}")
        self.text_to_speech(response)

if __name__ == "__main__":
    try:
        chatbot = HealthcareChatbot()
        chatbot.chat()
    except Exception as e:
        print(f"An error occurred: {str(e)}")