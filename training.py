import random
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


Lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ",", "."]


for intent in intents['intents']:
    for patterns in intent['patterns']:
        word_list = word_tokenize(patterns)
        words.extend(word_list)
        documents.append((word_list, intent['tags']))
        if intent['tags'] not in classes:
            classes.append(intent['tags'])
print(documents)
"***************************************************"

words = [Lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [Lemmatizer.lemmatize(word.lower())for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

train_x = [x[0] for x in training]
train_y = [x[1] for x in training]

train_x = np.array(train_x)
train_y = np.array(train_y)

model = tf.keras.models.Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chat_bot.keras')
print('done')
