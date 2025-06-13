import os
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

from model import UNet
from utils import SegmentationDataset, bce_dice_loss, compute_metrics

def load_dataset(image_dir, mask_dir, split=(0.7, 0.2, 0.1)):
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    dataset = SegmentationDataset(images, masks, transform=transforms.ToTensor())
    n_total = len(dataset)

    # Se tiver menos de 5, tudo para treino, val/teste vazios
    if n_total < 5:
        # Retorna listas para compatibilidade com random_split
        return [dataset, [], []]
    else:
        n_train = int(split[0] * n_total)
        n_val = int(split[1] * n_total)
        n_test = n_total - n_train - n_val
        splits = [n_train, n_val, n_test]
        # Corrige para nunca dar zero em nenhum split
        for i in range(3):
            if splits[i] == 0:
                splits[i] = 1
        while sum(splits) > n_total:
            for i in range(3):
                if splits[i] > 1 and sum(splits) > n_total:
                    splits[i] -= 1
        return random_split(dataset, splits)

def train(model, loader, optimizer, device):
    model.train()
    epoch_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = bce_dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader) if len(loader) > 0 else 0

def evaluate(model, loader, device):
    model.eval()
    metrics = []
    loss_total = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = bce_dice_loss(outputs, masks)
            m = compute_metrics(outputs, masks)
            m['loss'] = loss.item()
            metrics.append(m)
            loss_total += loss.item()
    if len(metrics) == 0:
        return {}
    avg = {k: sum(d[k] for d in metrics) / len(metrics) for k in metrics[0]}
    avg['loss'] = loss_total / len(metrics)
    return avg

def plot_gradcam(model, img, mask, save_path):
    img = img.unsqueeze(0)
    img.requires_grad = True
    output = model(img)
    cam = model.grad_cam('up4')
    heat = cam.squeeze().cpu().numpy()
    pred = output.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img.squeeze().permute(1, 2, 0).cpu())
    ax[0].set_title('Imagem')
    ax[1].imshow(pred, cmap='gray')
    ax[1].imshow(heat, alpha=0.5, cmap='jet')
    ax[1].set_title('Grad-CAM')
    ax[2].imshow(mask.squeeze().cpu(), cmap='gray')
    ax[2].set_title('Máscara')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # Exemplo simplificado de uso para Google Colab
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Usando dispositivo:', device)

    # Diretórios com imagens e máscaras
    img_dir = os.environ.get('IMG_DIR', 'imgs')
    mask_dir = os.environ.get('MASK_DIR', 'masks')

    train_ds, val_ds, test_ds = load_dataset(img_dir, mask_dir)
    
    # Garante que loaders não quebrem se algum split for vazio
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True) if len(train_ds) > 0 else None
    val_loader = DataLoader(val_ds, batch_size=4) if len(val_ds) > 0 else None
    test_loader = DataLoader(test_ds, batch_size=4) if len(test_ds) > 0 else None

    model = UNet(in_ch=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    for epoch in range(5):
        if train_loader is not None:
            loss = train(model, train_loader, optimizer, device)
        else:
            loss = 0
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            dice = val_metrics.get('dice', 0)
            scheduler.step(val_metrics['loss'] if 'loss' in val_metrics else loss)
        else:
            dice = 0
        print(f"Ep{epoch+1} Loss:{loss:.4f} Val Dice:{dice:.4f}")

    if test_loader is not None and len(test_ds) > 0:
        test_metrics = evaluate(model, test_loader, device)
        print('Metrics Teste:', test_metrics)
        # Gerar Grad-CAM para primeira imagem de teste
        img, mask = test_ds[0]
        plot_gradcam(model, img.to(device), mask.to(device), 'gradcam_example.png')
        print('Grad-CAM salvo em gradcam_example.png')
    else:
        print('Sem dados de teste para métricas ou Grad-CAM.')


