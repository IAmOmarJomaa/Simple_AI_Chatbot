# Simple AI Chatbot
**Project Overview**
This project is a fundamental AI Chatbot built to familiarize myself with the core concepts of Natural Language Processing (NLP) and the implementation of neural networks. It processes user input to understand their intent and provides relevant responses based on a predefined knowledge base.

**Technologies Used**

- This project leverages the following key technologies:

- Python: The primary programming language.

- NLTK (Natural Language Toolkit): For text preprocessing tasks such as tokenization and lemmatization.

- TensorFlow/Keras: For building, training, and deploying the neural network model that classifies user intents.

- JSON: Used as a simple database to define chatbot intents, patterns, and responses.

- Pickle: For serializing and deserializing Python objects (vocabulary and classes) for model persistence.

- NumPy: For numerical operations, especially in handling data arrays for the neural network.

**How It Works & What I Learned**
The chatbot operates in two main stages:

1. Training (training.py): This script reads patterns and responses from intents.json. It uses NLTK to tokenize and lemmatize words, creating a bag-of-words representation for each pattern. This numerical data then feeds into a simple feed-forward neural network built with TensorFlow/Keras. The model learns to map input patterns to their corresponding intent categories. This process was key to understanding data preparation for neural networks and basic model architecture (Dense layers, activation functions, dropout for regularization).

2. Inference (AI_chatbot.py): This script loads the trained model and other processed data. When a user inputs a message, it's preprocessed into a bag-of-words vector, which is then fed to the neural network for intent prediction. Based on the highest predicted intent, a random response is chosen from intents.json. Through this, I gained practical insight into how a trained model is used to make predictions and generate dynamic responses.

This project was an excellent hands-on exercise in understanding the end-to-end process of building a basic NLP application. Specifically, I gained:

- Natural Language Processing (NLP) Fundamentals: A practical understanding of tokenization and lemmatization as essential first steps in preparing text data. I also understood the concept and implementation of Bag-of-Words as a simple text representation.

- Understanding Bag-of-Words Limitations: I learned that the Bag-of-Words approach, while effective for simpler tasks, does not preserve word order. This means sentences with the same words but different meanings (e.g., "dog bites man" vs. "man bites dog") would have identical representations. This insight highlighted the need for more advanced techniques.

- Awareness of Advanced NLP Techniques: This limitation led me to discover Word Embeddings (like Word2Vec, GloVe), Recurrent Neural Networks (RNNs) (including LSTM and GRU), and Transformers (like BERT, GPT) as methods designed to capture word order and context, providing a deeper understanding of the evolution of NLP models.

- Neural Network Basics with Keras/TensorFlow: Learned how to construct a simple feed-forward neural network, apply dropout for regularization to combat overfitting, and understand the training process (loss functions, optimizers, metrics).

- Project Structure and Workflow: Experienced the typical machine learning project workflow: data preparation, model training, model saving/loading, and inference.

**Project Dates**
Start Date: 03/12/2024
Finish Date: 06/12/2024