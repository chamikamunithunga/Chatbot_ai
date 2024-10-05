# Step 1: Import Libraries
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 2: Load the Dataset
df = pd.read_csv('chatbot_data.csv')

# Step 3: Preprocess the Data
# Download necessary resources from nltk
nltk.download('punkt')
nltk.download('wordnet')

# Define a function to tokenize and lemmatize the text
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())  # Convert to lowercase and tokenize
    lemmatizer = nltk.WordNetLemmatizer()      # Initialize lemmatizer
    return [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize each token

# Apply the preprocess function to each question in the dataset
df['processed_question'] = df['question'].apply(preprocess)

# Step 4: Vectorize the Text
# Convert the questions to a matrix of token counts using CountVectorizer
vectorizer = CountVectorizer().fit(df['question'])
question_vectors = vectorizer.transform(df['question']).toarray()

# Step 5: Define a Function for Similarity Matching
def get_response(user_input):
    # Preprocess user input
    processed_input = ' '.join(preprocess(user_input))
    
    # Vectorize user input
    input_vector = vectorizer.transform([processed_input]).toarray()
    
    # Compute similarity scores
    similarity_scores = cosine_similarity(input_vector, question_vectors)
    
    # Get the index of the most similar question
    index = np.argmax(similarity_scores)
    
    # Return the corresponding answer
    return df['answer'].iloc[index]



# Step 6: Run the Chatbot
def chatbot():
    print("Chatbot: Hello! How can I assist you today? (type 'exit' to stop)")
    
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

# Step 7: Start the Chatbot
chatbot()
