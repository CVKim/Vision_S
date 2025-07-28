from sam2.build_sam import build_sam2
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch 
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            # Stage 1: (2, 2) -> (4, 4)
            nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Stage 2: (4, 4) -> (8, 8)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Stage 3: (8, 8) -> (16, 16)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Stage 4: (16, 16) -> (32, 32)
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)  # Final layer
        )

    def forward(self, x):
        return self.decoder(x)

class LinearProbingSam2(nn.Module):
    def __init__(self, in_channels, num_classes, sam2_checkpoint, model_cfg):
        super().__init__()

        model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
        print(model.image_encoder)
        self.encoder = model.image_encoder
        
        for param in self.encoder.parameters():
            param.requires_grad = False  # Frozen


        self.decoder = Decoder(in_channels, num_classes)

    def forward(self, x, labels=None, **kwargs):
        outputs = self.encoder(x)['vision_features']
        logits = self.decoder(outputs)
        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255)
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return {'loss': loss, 'logits': logits}