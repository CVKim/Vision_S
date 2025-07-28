import torch.nn as nn 

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes, in_channels, hid_channels):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hid_channels, hid_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(hid_channels*2, num_classes)
        
        
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x 
        
if __name__ == '__main__':
    import torch 
    
    classifier = SimpleClassifier(4, 3, 32)
    inputs = torch.randn((2, 3, 224, 224))
    
    outputs = classifier(inputs)