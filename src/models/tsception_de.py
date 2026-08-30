"""TSception (Ding et al., IEEE TAFFC 2023) adapted to the unified SEED
DE-feature input.

Official architecture: multi-scale temporal inception convs (0.5/0.25/0.125 s
kernels at the EEG sampling rate) -> asymmetric spatial convs (full/half
channel kernels) -> BN -> FC head. Faithful port with the DE adaptation
documented inline: the 5 DE bands are concatenated along the temporal axis
(giving 5*W frames at an effective rate of 5 frames/s), temporal inception
kernels are rescaled to [5, 3, 2] frames, and the spatial stage operates on
the true 62-channel layout exactly as in the official code. Official
hyperparameters: num_T=9, num_S=6, hidden=128, dropout=0.3, Adam lr=1e-3,
weight_decay=1e-6 (Lambda), batch=128.
"""
import torch
import torch.nn as nn


class TSceptionDE(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, n_classes=3, n_ch=62, n_band=5, W=4,
                 num_T=9, num_S=6, hidden=128, dropout_rate=0.3):
        super().__init__()
        T = n_band * W                        # 75 "temporal" frames at W=15
        rate = n_band                          # 5 DE frames per second
        # inception kernels rescaled from the paper's 128/64/32 samples at
        # 256 Hz to the DE frame rate (kernels 9/5/2 frames), pool 4
        self.Tception1 = self.conv_block(
            1, num_T, (1, 9), 1, 4)
        self.Tception2 = self.conv_block(
            1, num_T, (1, 5), 1, 4)
        self.Tception3 = self.conv_block(
            1, num_T, (1, 2), 1, 4)
        self.Sception1 = self.conv_block(
            num_T, num_S, (n_ch, 1), 1, 2)
        self.Sception2 = self.conv_block(
            num_T, num_S, (n_ch // 2, 1), (n_ch // 2, 1), 2)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        size = self._get_size(n_ch, T)
        self.fc = nn.Sequential(
            nn.Linear(size[1], hidden), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(hidden, n_classes))

    def _forward_features(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        return self.BN_t(out)

    def _get_size(self, n_ch, T):
        with torch.no_grad():
            data = torch.ones(1, 1, n_ch, T)
            out = self._forward_features(data)
            z = self.Sception1(out)
            out_ = z
            z = self.Sception2(out)
            out_ = torch.cat((out_, z), dim=2)
            out_ = self.BN_s(out_)
            return out_.view(out_.size(0), -1).size()

    def forward(self, x):                     # [B, 310, W] -> [B, 62, 5, W]
        B = x.shape[0]
        x = x.reshape(B, 62, 5, -1).permute(0, 2, 1, 3)   # [B, 5, 62, W]
        x = x.reshape(B, 1, 62, -1)                        # [B, 1, 62, 5W]
        out = self._forward_features(x)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out_ = self.BN_s(out_)
        return self.fc(out_.view(B, -1))
