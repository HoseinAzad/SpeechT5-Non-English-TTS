import torch
from transformers import SpeechT5ForTextToSpeech
from functools import partial

class T5TTS():

    def __init__(self, check_point, vocab_size):
        self.model = SpeechT5ForTextToSpeech.from_pretrained(check_point)
        self.model.resize_token_embeddings(vocab_size)
        self.model.config.use_cache = False
        # set language and task for generation and re-enable cache
        self.model.generate = partial(self.model.generate, use_cache=True)

    def get(self, device, weights_path= None):
        model =  self.model.to(device)
        if weights_path!= None:
            model.load_state_dict(torch.load(weights_path))
        return model
    

def get_model(model_checkpoint, vocab_size, device):
    t5tts = T5TTS(model_checkpoint, vocab_size)
    model = t5tts.get(device)
    return model