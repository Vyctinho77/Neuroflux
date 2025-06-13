import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Duas camadas de convolução consecutivas com BatchNorm e ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Redução de escala com maxpool e depois conversão dupla."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.model(x)

class Up(nn.Module):
    """Aumento de escala e conversão dupla."""
    def __init__(self, in_ch_from_up, in_ch_from_skip, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch_from_up, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + in_ch_from_skip, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """U-Net customizada para múltiplas modalidades."""
    def __init__(self, in_ch=2, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512, 512, 256)
        self.up2 = Up(256, 256, 128)
        self.up3 = Up(128, 128, 64)
        self.up4 = Up(64, 64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # Para ganchos Grad-CAM
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output
            return hook
        self.up1.conv.register_forward_hook(get_activation('up1'))
        self.up2.conv.register_forward_hook(get_activation('up2'))
        self.up3.conv.register_forward_hook(get_activation('up3'))
        self.up4.conv.register_forward_hook(get_activation('up4'))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)

    def grad_cam(self, target_layer='up4', target_class=None):
        """Gerar Grad-CAM para a camada decodificadora escolhida."""
        activations = self.activations[target_layer]
        grads = torch.autograd.grad(
            outputs=activations,
            inputs=activations,
            grad_outputs=torch.ones_like(activations),
            retain_graph=True,
            create_graph=True
        )[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(activations.size(2), activations.size(3)), mode='bilinear', align_corners=False)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam
