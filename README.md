# SpeechT5-Non-English-TTS
Fine-tune SpeechT5 for non-English text-to-speech tasks, implemented in PyTorch.

This repository contains code and resources for fine-tuning (or training) a SpeechT5 model on a non-English language for a text-to-speech task. The project leverages Huggingface's `transformers` library and `speechbrain` to load necessary models and tools. Other parts of the code, such as data preprocessing and train and evaluate functions, have been fully implemented using PyTorch. Therefore, feel free to make any changes you need to train your model efficiently.

## Project Overview
The main objective of this project is to fine-tune the SpeechT5 model for text-to-speech on a non-English language. The steps include:
1. Setting up the environment.
2. Loading necessary tools (tokenizer and feature extractor) and models (SpeechT5 itself, a model to generate X-vector speaker embeddings, and the vocoder).
3. **Most importantly:** Adding the unique characters of the language you want to fine-tune the model on to the tokenizer and modifying the input embedding matrix of the model accordingly.
4. Loading and preprocessing your data.
5. Training and evaluating the model.

## Generated Samples
Here are some generated samples from the model that I trained on the Persian Common Voice dataset.




**Sample 1**
https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/assets/74851226/0447ab14-a68d-4a5e-94db-5ddd29e111d3
[1.webm](https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/assets/74851226/dfd5676d-f0f4-4f35-86f6-9748a16a62be)


**Sample 2**
<audio controls>
  <source src="https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/blob/master/results/2.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

**Sample 3**
<audio controls>
  <source src="https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/blob/master/results/3.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

**Sample 4**
<audio controls>
  <source src="https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/blob/master/results/4.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

**Sample 5**
<audio controls>
  <source src="https://github.com/HoseinAzad/SpeechT5-Non-English-TTS/blob/master/results/5.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

## References 
This code draws lessons from:<br>
https://huggingface.co/learn/audio-course/en/chapter6/fine-tuning
