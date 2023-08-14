from tqdm import tqdm
import torch

from torch.utils.tensorboard import SummaryWriter
from simple_tokenizer import load_and_transform_text

import config as CFG
from model import CLIP
from utils import AvgMeter, get_lr

from collections import OrderedDict


CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def order_batch(labels):
    # Create positive and negative pairings of images and text labels

    unique_labels = OrderedDict()
    label_idx = {}
    counter = 0
    for i, l in enumerate(labels):
        if l not in unique_labels:
            unique_labels[l] = i
            label_idx[l] = counter
            counter += 1

    labels_matrix = torch.zeros(len(labels), len(unique_labels))
    for i, l in enumerate(labels):
        labels_matrix[i][label_idx[l]] = 1

    return labels_matrix, torch.Tensor(list(unique_labels.values())).long()


def create_pairs(images, text, tokens):
    unique_labels = {}
    image_list = []
    token_list = []
    labels = []
    for i, l in enumerate(text):
        if l not in unique_labels:
            unique_labels[l] = tokens[i]
    
    for i, img in enumerate(images):
        for l in unique_labels:
            image_list.append(img)
            token_list.append(unique_labels[l])
            if l == text[i]:
                labels.append(1) # Positive pair
            else:
                labels.append(0)  # Negative pair

    return torch.stack(image_list, dim=0), torch.stack(token_list, dim=0), torch.Tensor(labels)


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for img, text in tqdm_object:
        text = [CLASS_NAMES[i] for i in text]
        tokens = load_and_transform_text(text, CFG.device)  # Tokenizer
        img, tokens, labels = create_pairs(img, text, tokens)
        img = img.to(CFG.device)
        labels = labels.unsqueeze(1).to(CFG.device)
        tokens = tokens.to(CFG.device)
        loss, _ = model(img, tokens, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = img.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter.avg


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for img, text in tqdm_object:
        text = [CLASS_NAMES[i] for i in text]
        tokens = load_and_transform_text(text, CFG.device)  # Tokenizer
        img, tokens, labels = create_pairs(img, text, tokens)
        img = img.to(CFG.device)
        labels = labels.unsqueeze(1).to(CFG.device)
        tokens = tokens.to(CFG.device)
        loss, _ = model(img, tokens, labels)

        count = img.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter.avg


def main():
    writer = SummaryWriter("CIFAR_10_training_logs")
    from dataloaders import get_original_cifar10_dataloaders
    root = '/data/datasets'
    train_loader, valid_loader = get_original_cifar10_dataloaders(
        root, train_bsz=CFG.batch_size, val_bsz=CFG.batch_size
    )
    model = CLIP(
        embed_dim=CFG.projection_dim,
        image_resolution=CFG.size,
        vision_layers=3,
        vision_width=512,
        vision_patch_size=4,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=8,
        transformer_layers=3
    )
    model = model.to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.epochs)
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        writer.add_scalar("Train loss", train_loss, epoch + 1)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            writer.add_scalar("Val loss", valid_loss, epoch + 1)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "best_model_CIFAR10.pt")
            print("Saved Best Model!")


if __name__ == "__main__":
    main()
