import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from model import CNN_RNN_Model
from dataloader import get_dataloader

import warnings
warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter

def buid_dataloader(type="train"):
    if type == "train":
        transform = transforms.Compose([
            transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.1),  # Slight perspective changes
            # transforms.RandomResizedCrop(250, scale=(0.8, 1.0)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.2)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandAugment(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        csv_dir = 'train.json'
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        csv_dir = 'test.json'

    root_dir = 'wikiart/'

    if type == "train":
        return get_dataloader(csv_dir, root_dir, transform, 32)
    return get_dataloader(csv_dir, root_dir, transform, 32)


def train_one_epoch(model, train_loader, optimizer, loss_fn, iteration, summary_writer, device):
    model.train()
    losses = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, data in progress_bar:
        images, labels = data
        images = images.to(device)
        labels = torch.tensor(labels, device=device)

        optimizer.zero_grad()
        style_pred, artist_pred, genre_pred = model(images)
        artist_gt = labels[:, 0]
        style_gt = labels[:, 1]
        genre_gt = labels[:, 2]

        artist_gt = torch.nn.functional.one_hot(artist_gt, num_classes=23).float()
        style_gt = torch.nn.functional.one_hot(style_gt, num_classes=27).float()
        genre_gt = torch.nn.functional.one_hot(genre_gt, num_classes=10).float()

        loss_style = loss_fn(style_pred, style_gt.to(device))
        loss_artist = loss_fn(artist_pred, artist_gt.to(device))
        loss_genre = loss_fn(genre_pred, genre_gt.to(device))
        loss = loss_style + loss_artist + loss_genre
        summary_writer.add_scalar("Loss/train", loss.item(), iteration)
        summary_writer.add_scalar("Style Loss/train", loss_style.item(), iteration)
        summary_writer.add_scalar("Artist Loss/train", loss_artist.item(), iteration)
        summary_writer.add_scalar("Genre Loss/train", loss_genre.item(), iteration)
        iteration += 1
        losses += loss.item()
        loss.backward()
        optimizer.step()
        # count += 1
        progress_bar.set_description(f"Loss: {losses / len(train_loader)}")
        # if count > 10:
        #     break
    return losses / len(train_loader), model, iteration

def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    losses = 0
    corrects_style = 0
    corrects_artist = 0
    corrects_genre = 0
    count = 0
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
    for batch_idx, data in progress_bar:
        images, labels = data
        images = images.to(device)
        labels = torch.tensor(labels, device=device)

        with torch.no_grad():
            style_pred, artist_pred, genre_pred = model(images)
            artist_gt = labels[:, 0]
            style_gt = labels[:, 1]
            genre_gt = labels[:, 2]

            artist_gt = torch.nn.functional.one_hot(artist_gt, num_classes=23).float()
            style_gt = torch.nn.functional.one_hot(style_gt, num_classes=27).float()
            genre_gt = torch.nn.functional.one_hot(genre_gt, num_classes=10).float()

            loss_style = loss_fn(style_pred, style_gt)
            loss_artist = loss_fn(artist_pred, artist_gt)
            loss_genre = loss_fn(genre_pred, genre_gt)
            loss = loss_style + loss_artist + loss_genre
            # check accuracy
            style_pred = torch.round(style_pred)
            artist_pred = torch.round(artist_pred)
            genre_pred = torch.round(genre_pred)
            corrects_style += (torch.argmax(style_gt, dim=1) == torch.argmax(style_pred.sigmoid(), dim=1)).sum().item()
            corrects_artist += (torch.argmax(artist_gt, dim=1) == torch.argmax(artist_pred.sigmoid(), dim=1)).sum().item()
            corrects_genre += (torch.argmax(genre_gt, dim=1) == torch.argmax(genre_pred.sigmoid(), dim=1)).sum().item()
            losses += loss.item()
            # count += 1
            progress_bar.set_description(f"Loss: {losses /len(val_loader)} Style: {corrects_style / len(val_loader.dataset)} "
                                 f"Artist: {corrects_artist / len(val_loader.dataset)} Genre: {corrects_genre / len(val_loader.dataset)}")

    return (losses / len(val_loader), corrects_artist / len(val_loader.dataset),
            corrects_genre / len(val_loader.dataset), corrects_style / len(val_loader.dataset), model)

def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10):
    best_acc = -float("inf")
    summary_writer = SummaryWriter()
    iteration = 0
    # val_loss, artist_acc, genre_acc, style_acc, model = evaluate(model, val_loader, loss_fn, device)
    # print(f"Val loss: {val_loss}")
    # print(f"Artist accuracy: {artist_acc}")
    # print(f"Genre accuracy: {genre_acc}")
    # print(f"Style accuracy: {style_acc}")
    # best_acc = artist_acc + genre_acc + style_acc

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, model, iteration = train_one_epoch(model, train_loader, optimizer, loss_fn, iteration, summary_writer, device)
        print(f"Train loss: {train_loss}")
        val_loss, artist_acc, genre_acc, style_acc, model = evaluate(model, val_loader, loss_fn, device)
        print(f"Val loss: {val_loss}")
        print(f"Artist accuracy: {artist_acc}")
        print(f"Genre accuracy: {genre_acc}")
        print(f"Style accuracy: {style_acc}")
        val_acc = artist_acc + genre_acc + style_acc

        summary_writer.add_scalar("Artist Accuracy/val", artist_acc, epoch)
        summary_writer.add_scalar("Genre Accuracy/val", genre_acc, epoch)
        summary_writer.add_scalar("Style Accuracy/val", style_acc, epoch)
        summary_writer.add_scalar("Total Accuracy/val", val_acc, epoch)
        summary_writer.add_scalar("Loss/val", val_loss, epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")

def main():
    train_loader = buid_dataloader(type="train")
    val_loader = buid_dataloader(type="test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    num_classes_style = 27  # Number of style classes
    num_classes_artist = 23  # Number of artist classes
    num_classes_genre = 10
    # num_classes_style = 1  # Number of style classes
    # num_classes_artist = 1  # Number of artist classes
    # num_classes_genre = 1  # Number of genre classes
    cnn_hidden_size = 2048  # Output size of the CNN feature extractor
    rnn_hidden_size = 512  # Hidden size of the RNN

    # Initialize the model
    model = CNN_RNN_Model(num_classes_style, num_classes_artist, num_classes_genre, cnn_hidden_size, rnn_hidden_size).to(device=device)
    # load model if file exists
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
        print("Loaded model")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=50)


if __name__ == "__main__":
    main()