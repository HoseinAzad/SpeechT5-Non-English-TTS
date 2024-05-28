import matplotlib.pyplot as plt
import torch


def save_model(self, model, path):
    torch.save(model.state_dict(), path)


def save_checkpoint(model, optimizer, scheduler, epoch, minloss, save_path):
    checkpoint = {
        'minloss': minloss,
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler}
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint):
    minloss = checkpoint['minloss']
    epoch = checkpoint['epoch']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    return model, optimizer, scheduler, epoch, minloss


def get_persian_tokens():
    tokens = [char for char in " ابپتسجچخعدزرژش‌فقکآگلمنوهی"]
    return tokens


def plotLoss(loss_list, title):
    plt.figure(figsize=(10, 4))
    plt.plot(loss_list[:, 0], label="Train_loss")
    plt.plot(loss_list[:, 1], label="Validation_loss")
    plt.set_title("Loss Curves - " + title, fontsize=12)
    plt.set_ylabel("Loss", fontsize=10)
    plt.set_xlabel("Epoch", fontsize=10)
    plt.legend(prop={'size': 10})
    plt.grid()
    plt.show()
