# SpeechT5-Non-English-TTS
Fine-tune SpeechT5 for non-English text-to-speech tasks, implemented in PyTorch.

![speecht5_framework](https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/assets/74851226/34589034-7790-4a5a-a037-6ee0df4d0a96)

This repository contains code and resources for fine-tuning (or training) a SpeechT5 model on a non-English language for a text-to-speech task. The project leverages Huggingface's `transformers` library and `speechbrain` to load necessary models and tools. Other parts of the code, such as data preprocessing and train and evaluate functions, have been fully implemented using PyTorch. Therefore, feel free to make any changes you need to train your model efficiently.

<br>

## Project Overview
The main objective of this project is to fine-tune the SpeechT5 model for text-to-speech on a non-English language. The steps include:
1. Setting up the environment.
2. Loading necessary tools (tokenizer and feature extractor) and models (SpeechT5 itself, a model to generate X-vector speaker embeddings, and the vocoder).
3. **Most importantly:** Adding the unique characters of the language you want to fine-tune the model on to the tokenizer and modifying the input embedding matrix of the model accordingly.
4. Loading and preprocessing your data.
5. Training and evaluating the model.

<br>

## Generated Samples
Here are some generated samples from the model that I trained on the Persian Common Voice dataset.

**Sample 1**

https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/assets/74851226/6f24e01d-aa9f-47d1-b033-22da3eb8bf87

**Sample 2**

https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/assets/74851226/428a06be-c63e-4d2c-9a31-d69380eaa312

**Sample 3**

https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/assets/74851226/18273f85-a6a6-44d9-8a17-10aac2e67838


**Sample 4**

https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/assets/74851226/d90dd225-253f-4b19-81a0-b8fff3da1086

**Sample 5**

https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/assets/74851226/3567ad20-4784-4bd0-832b-ede91b544aea

<br>

## References 
This code draws lessons from:<br>
https://huggingface.co/learn/audio-course/en/chapter6/fine-tuning
