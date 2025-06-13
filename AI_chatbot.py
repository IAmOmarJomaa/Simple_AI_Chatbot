import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chat_bot.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.tokenize.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD =0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list
def get_response(intens_list, intens_json):
    tags = intens_list[0]['intent']
    list_of_intents = intens_json['intents']
    result = None  # Initialize result to handle no match case
    for i in list_of_intents:
        if i['tags'] == tags:
            result = random.choice(i['responses'])
            break
    return result  # Return result after the loop

print('go bot is ready')

while True:
    message = input("You: ")
    ints = predict_class(message)
    print("Predicted Intents:", ints)  # Debugging
    if not ints:
        print("No intent detected.")  # Handle empty intent list
        continue
    res = get_response(ints, intents)
    print("Response:", res)