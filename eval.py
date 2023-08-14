import torch
from tqdm import tqdm

import config as CFG
from cross_clip_model import CLIP
from simple_tokenizer import load_and_transform_text


def create_test_pairs(images, text, tokens, class_names):
    image_list = []
    token_list = []
    labels = []
    
    for i, img in enumerate(images):
        for j, l in enumerate(class_names):
            image_list.append(img)
            token_list.append(tokens[j])
            if l == text[i]:
                labels.append(1) # Positive pair
            else:
                labels.append(0)  # Negative pair

    return torch.stack(image_list, dim=0), torch.stack(token_list, dim=0), torch.Tensor(labels)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    val_bsz = train_bsz = 1

    model_path = "best_model_C_4_blocks_CIFAR100.pt" #"best_model_C_100_epochs_CIFAR100.pt"

    model = CLIP(
        embed_dim=CFG.projection_dim,
        image_resolution=CFG.size,
        vision_layers=4,
        vision_width=512,
        vision_patch_size=4,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=8,
        transformer_layers=4
    )
    model = model.to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    # Prepare data and training entities
    from dataloaders import get_original_cifar100_dataloaders
    root = "/data/datasets"
    _, test_loader = get_original_cifar100_dataloaders(
        root, train_bsz=train_bsz, val_bsz=val_bsz
    )
    #class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_names = []
    with open('cifar_100_cls.txt', 'r') as f:
        for row in f:
            class_names.append(row.strip())

    tokens = load_and_transform_text(class_names, CFG.device)  # Tokenizer
    correct = 0
    total_size = 0
    with torch.no_grad():
        for img, text in tqdm(test_loader):
            img = img.to(CFG.device)
            prob_min = 0
            best_class = 'airplane'
            for i, c in enumerate(class_names):
                t = tokens[i].unsqueeze(0)
                text = text.unsqueeze(1)
                _, pred = model(img, t)
                if pred > prob_min:
                    prob_min = pred
                    best_class = c
            #_, preds = model(img, tokens, labels)
            #preds = torch.ones(labels.shape).cuda()
            #preds[preds <= 0.5] = 0
            #preds[preds > 0.5] = 1
            #correct += preds.eq(labels.view_as(preds)).sum().item()
            #total_size += labels.shape[0]
            if best_class == class_names[text]:
                correct += 1
    accuracy = 100.0 * correct / len(test_loader.dataset) #total_size
    print(f"accuracy: {accuracy:.01f}" + "\n")

if __name__ == "__main__":
    main()
