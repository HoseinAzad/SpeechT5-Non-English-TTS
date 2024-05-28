SPEECHT5_CHP = "microsoft/speecht5_tts"
VOCODER_CHP = "microsoft/speecht5_hifigan"
BEST_MODEL_PATH = '/content/drive/MyDrive/SpeechT5/best_model.pt'
DRIVE_CHP_PATH = '/content/drive/MyDrive/SpeechT5/checkpoint.pth'

class Configs():

    def __init__(self, seed,
                 finetune_cp_path = DRIVE_CHP_PATH,
                 model_checkpoint= SPEECHT5_CHP,
                 vocoder_checkpoint = VOCODER_CHP,
                 learning_rate= 1e-5,
                 train_batch_szie= 18,
                 weight_decay= 1e-4,
                 best_model_path = BEST_MODEL_PATH
                 ):

        self.seed = seed
        self.finetune_cp_path = finetune_cp_path
        self.model_checkpoint = model_checkpoint
        self.vocoder_checkpoint = vocoder_checkpoint
        self.learning_rate = learning_rate
        self.train_batch_szie = train_batch_szie
        self.weight_decay = weight_decay
        self.best_model_path = best_model_path