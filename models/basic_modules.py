import torch
from torch import nn
from torch.nn import init

# https://github.com/sxhxliang/BigGAN-pytorch/blob/4cbad24f7b49bf55f2b1b6aa8451b2db495b707c/model_resnet.py


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.query_conv)
        init_conv(self.key_conv)
        init_conv(self.value_conv)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        # B X CX(N)
        proj_query = \
            self.query_conv(x) \
                .view(m_batchsize, -1, width*height) \
                .permute(0, 2, 1)
        # B X C x (*W*H)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        # transpose check
        energy = torch.bmm(proj_query, proj_key)
        # BX (N) X (N)
        attention = self.softmax(energy)
        # B X C X N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_embed = nn.Embedding(num_classes, num_features)
        nn.init.normal_(self.gamma_embed.weight, 1, 0.02)
        self.beta_embed = nn.Embedding(num_classes, num_features)
        nn.init.zeros_(self.beta_embed.weight)

    def forward(self, x, y):
        return \
            self.gamma_embed(y).view(-1, self.num_features, 1, 1) \
            * self.bn(x) \
            + self.beta_embed(y).view(-1, self.num_features, 1, 1)
