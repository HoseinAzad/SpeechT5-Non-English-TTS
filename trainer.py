import os
import random
import numpy as np
from IPython.display import Audio, Markdown

import torch
from transformers import SpeechT5HifiGan
from transformers import SpeechT5FeatureExtractor, SpeechT5Tokenizer
from speechbrain.pretrained import EncoderClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import *
from model import get_model
from dataset import get_data_loaders
from utils import get_persian_tokens, load_checkpoint, save_checkpoint


class Trainer():

    def __init__ (self, config, dataset, checkpoint=None):
        
        self.dataset = dataset
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load SpeechT5 Tokenizer (to process text) and FeatureExtractor (to process audio)
        print('Loading Tokenizer and FeatureExtractor ...', end = '')
        self.feature_extractor = SpeechT5FeatureExtractor.from_pretrained(config.model_checkpoint)
        self.tokenizer = SpeechT5Tokenizer.from_pretrained(config.model_checkpoint)
        print('\rTokenizer and FeatureExtractor loaded successfully.'); print('-' * 50)

        # Update tokenizer with persian character
        self.tokenizer.add_tokens(get_persian_tokens())

        # Load vocoder (hifigan) to generate waveform from specogram
        print('Loading Vocoder ...', end = '')
        self.vocoder = SpeechT5HifiGan.from_pretrained(config.vocoder_checkpoint).to(self.device)
        print('\rSpeechT5 loaded successfully.'); print('-' * 50)


        # Load speaker model (to generate speaker embedding)
        print('Loading speaker model ...', end = '')
        spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
        self.speaker_model = EncoderClassifier.from_hparams(source = spk_model_name,
                                                            run_opts ={"device": self.device},
                                                            savedir = os.path.join("/tmp", spk_model_name))
        print('\rSpeaker model loaded successfully'); print('-'*50)

        if checkpoint == None:
            # Load text to speech model
            print('Loading SpeechT5 model ...', end = '')
            self.t5tts = get_model(config.model_checkpoint, len(self.tokenizer), self.device)
            print('\rSpeechT5 loaded successfully.'); print('-' * 50)

            # Initialize the Optimizer and Scheduler
            self.optimizer = torch.optim.AdamW(self.model.parameters(), config.learning_rate, weight_decay = config.weight_decay)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.6, patience= 3, min_lr=1e-6, threshold=0.01)
            self.epoch, self.minloss= 0, float('inf')
        else:
            self.model, self.optimizer, self.scheduler, self.epoch, self.minloss = load_checkpoint(checkpoint)
            print('\rCheckpoint loaded successfully'); print('-' * 50)

        # Create dataloaders
        self.train_dataloader, self.test_dataloader = get_data_loaders(dataset['train_data'],
                                                                      dataset['test_data'],
                                                                      config.train_batch_szie,
                                                                      self.tokenizer,
                                                                      self.feature_extractor,
                                                                      self.speaker_model,
                                                                      self.model.config.reduction_factor,)

        print('\nNumber of train batches =', len(self.train_dataloader), f'(batch size = {config.train_batch_szie})')
        print('Number of test batches =', len(self.test_dataloader), '(batch size = 2)'); print('-' * 50)



    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)


    def train (self, model, dataloader, optimizer, epoch, device):
        model.train()
        losses = []
        for step, batch in enumerate(dataloader):
            model.zero_grad()
            (input_ids, labels, speaker_embeddings, attnetion_mask) = [t.to(device) for t in batch[1:5]]

            loss, spectrogram = model(input_ids= input_ids,
                            labels= labels,
                            speaker_embeddings= speaker_embeddings,
                            attention_mask= attnetion_mask,
                            use_cache=False)[:2]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if step % 10 == 0:
                print(f'\r[Epoch][Batch] = [{epoch + 1}][{step}] -> Loss = {np.mean(losses):.4f}', end='')

        return np.mean(losses)



    def evaluate (self, model, dataloader, device):
        print(" / Evaluating ...", end='')
        model.eval()
        losses = []
        for step, batch in enumerate(dataloader):
            (input_ids, labels, speaker_embeddings, attnetion_mask) = [t.to(device) for t in batch[1:5]]
            loss, spectrogram = model(input_ids= input_ids,
                            labels= labels,
                            speaker_embeddings= speaker_embeddings,
                            attention_mask= attnetion_mask,
                            use_cache=False)[:2]
            losses.append(loss.item())

        return np.mean(losses)


    def generate(self, model, vocoder, train_dataloader, test_dataloader, device, epoch, sr=16e3):
        """
        This function is specifically designed to be run in an IPython notebook environment as it uses the `display` and 
        `Markdown` functions to show the generated speech outputs and corresponding text. 
        """
        # display(Markdown('## '+ f'Generated speeches at epoch {epoch}'))
        for (dataloader, status) in zip([train_dataloader, test_dataloader], ['train_data', 'test_data']):
            # display(Markdown('### '+ f'{status}:'))

            batch = next(iter(dataloader))
            for i in range(4):
                (input_ids, labels, speaker_embeddings, attnetion_mask) = [t[i].unsqueeze(0).to(device) for t in batch[1:5]]
                text = batch[0][i]
                output = model.generate_speech(input_ids, speaker_embeddings, vocoder=vocoder)
                audio = Audio(output.cpu().numpy(), rate=sr)
                # display(Markdown('#### '+ f'for: {text}'))
                # display(audio)


    def train_and_evaluate(self, n_epochs):

        loss_list =  []

        print('\rStart Training ....', end = '')
        for epoch in range(self.epoch, n_epochs):

            # Generate some examples
            # self.generate(self.model, self.vocoder, self.train_dataloader, self.test_dataloader,
            #               self.device, epoch)

            train_loss = self.train(self.model, self.train_dataloader, self.optimizer, epoch, self.device)
            test_loss = self.evaluate(self.model, self.test_dataloader, self.device)

            # Save the best model based on minimum loss
            if test_loss < self.minloss:
                self.minloss = test_loss
                self.utils.save_model(self.model, self.config.best_model_path, epoch)

            pre_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(train_loss)

            print(pre_lr)
            print(self.optimizer.param_groups[0]['lr'])

            loss_list.append([train_loss, test_loss])

            print(f'\r[{epoch+1}]--> (Train) Loss: {train_loss:.4f} | (VAl) Loss: {test_loss:.4f}')
            if pre_lr != self.optimizer.param_groups[0]['lr'] : print(f'Learning rate reduced at epoch {epoch + 1}!')

            save_checkpoint(self.model, self.optimizer, self.scheduler, epoch+1, self.minloss, self.config.finetune_checkpoint )

        # Load the best model (the state that model achieved minimum validation loss)
        return self.t5tts.get_model(self.device, weights_path= self.config.best_model_path)
