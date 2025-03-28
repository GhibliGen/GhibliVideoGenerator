import torch
import torch.nn as nn
import torch.nn.functional as F

class GhibliStyleNet(nn.Module):
    def __init__(self, params: dict):
        super(GhibliStyleNet, self).__init__()
        self.layers = params["layers"]
        self.channels = params["channels"]
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.channels, 4, 2, 1),
            nn.InstanceNorm2d(self.channels),
            nn.LeakyReLU(0.2),
            *[self._make_encoder_block(i) for i in range(self.layers)]
        )
        
        # Style Transformer
        self.transformer = nn.ModuleList([
            GhibliAttentionBlock(self.channels * 2) 
            for _ in range(4)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            *[self._make_decoder_block(i) for i in range(self.layers)],
            nn.ConvTranspose2d(self.channels, 3, 4, 2, 1),
            nn.Tanh()
        )
        
    def _make_encoder_block(self, idx: int) -> nn.Module:
        """Create encoder block with residual connections"""
        return nn.Sequential(
            nn.Conv2d(self.channels * (idx + 1), self.channels * (idx + 2), 4, 2, 1),
            nn.InstanceNorm2d(self.channels * (idx + 2)),
            nn.LeakyReLU(0.2)
        )
        
    def _make_decoder_block(self, idx: int) -> nn.Module:
        """Create decoder block with skip connections"""
        return nn.Sequential(
            nn.ConvTranspose2d(self.channels * (self.layers - idx), 
                             self.channels * (self.layers - idx - 1), 
                             4, 2, 1),
            nn.InstanceNorm2d(self.channels * (self.layers - idx - 1)),
            nn.ReLU()
        )
        
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Generate Ghibli-styled output"""
        features = self.encoder(x)
        styled = features
        
        for t_block in self.transformer:
            styled = t_block(styled)
            
        return self.decoder(styled)

class GhibliAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super(GhibliAttentionBlock, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, ch, h, w = x.size()
        q = self.query(x).view(batch, -1, h * w)
        k = self.key(x).view(batch, -1, h * w)
        v = self.value(x).view(batch, -1, h * w)
        
        attention = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)
        out = torch.bmm(v, attention.transpose(1, 2))
        out = out.view(batch, ch, h, w)
        
        return x + self.gamma * out
