import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable



def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.fc = deconv(z_dim, conv_dim*8, int(image_size/16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)
        
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)      # If image_size is 64, output shape is as below.
        out = self.fc(z)                            # (?, 512, 4, 4)
        out = F.leaky_relu(self.deconv1(out), 0.2)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.deconv2(out), 0.2)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.deconv3(out), 0.2)  # (?, 64, 32, 32)
        out = F.sigmoid(self.deconv4(out))             # (?, 3, 64, 64)
        return out
    
class Encoder(nn.Module):
    """Encoder (x -> z)."""
    def __init__(self, image_size=64, z_dim=100, conv_dim=64):
        super(Encoder, self).__init__()
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.fc = conv(conv_dim*8, z_dim*2, int(image_size/16), 1, 0, False)
    
    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = Variable(torch.randn(mu.size(0), mu.size(1)).cuda())
        z = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return z
                     
    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)  # (?, 64, 32, 32)
        h = F.leaky_relu(self.conv2(h), 0.2)  # (?, 128, 16, 16)
        h = F.leaky_relu(self.conv3(h), 0.2)  # (?, 256, 8, 8)
        h = F.leaky_relu(self.conv4(h), 0.2)  # (?, 512, 4, 4)
        h = self.fc(h)                        # (?, z_dim*2)
        
        h = h.view(h.size(0), h.size(1))
        mu, log_var = torch.chunk(h, 2, dim=1)  # mean and log variance.
        z = self.reparametrize(mu, log_var)
        return z, mu, log_var
    
    
class Discriminator(nn.Module):
    def __init__(self, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.fc = conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)
        
    def forward(self, x):                          # If image_size is 64, output shape is as below.
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 4, 4)
        out = self.fc(out).squeeze()
        return out