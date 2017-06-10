import torch.nn as nn 
import torch.nn.functional as F


class G(nn.Module):
    """Generator."""
    def __init__(self, image_size=64, z_dim=20, conv_dim=64):
        super(G, self).__init__()
        self.fc = nn.ConvTranspose2d(z_dim, conv_dim*8, 4, 1, 0)
        self.deconv1 = nn.ConvTranspose2d(conv_dim*8, conv_dim*4, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(conv_dim*4)
        self.deconv2 = nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(conv_dim*2)
        self.deconv3 = nn.ConvTranspose2d(conv_dim*2, conv_dim, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(conv_dim)
        self.deconv4 = nn.ConvTranspose2d(conv_dim, 3, 4, 2, 1)
        
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.fc(z)                                        # (?, 512, 4, 4)
        h = F.leaky_relu(self.bn1(self.deconv1(h)), 0.05)     # (?, 256, 8, 8)
        h = F.leaky_relu(self.bn2(self.deconv2(h)), 0.05)     # (?, 128, 16, 16)
        h = F.leaky_relu(self.bn3(self.deconv3(h)), 0.05)     # (?, 64, 32, 32)
        out = F.tanh(self.deconv4(h))                         # (?, 3, 64, 64)
        return out

    
class E(nn.Module):
    """Encoder."""
    def __init__(self, image_size=64, z_dim=20, num_classes=5, conv_dim=64):
        super(E, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, conv_dim, 4, 2, 1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1) 
        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1) 
        self.conv4 = nn.Conv2d(conv_dim*4, conv_dim*8, 4, 2, 1) 
        self.fc = nn.Conv2d(conv_dim*8, z_dim, 4, 1, 0)              
        
    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 32, 32)
        h = F.leaky_relu(self.conv2(h), 0.05)    # (?, 128, 16, 16)
        h = F.leaky_relu(self.conv3(h), 0.05)    # (?, 256, 8, 8)
        h = F.leaky_relu(self.conv4(h), 0.05)    # (?, 512, 4, 4)
        z = self.fc(h)                         # (?, z_dim, 1, 1)
        z = z.squeeze()
        #z[:, -self.num_classes:] = F.softmax(z[:, -self.num_classes:].clone())
        return z
    
    
class D(nn.Module):
    """Discriminator."""
    def __init__(self, image_size=64, num_classes=5, conv_dim=64):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(3, conv_dim, 4, 2, 1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1) 
        self.bn2 = nn.BatchNorm2d(conv_dim*2)
        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1) 
        self.bn3 = nn.BatchNorm2d(conv_dim*4)
        self.conv4 = nn.Conv2d(conv_dim*4, conv_dim*8, 4, 2, 1) 
        self.bn4 = nn.BatchNorm2d(conv_dim*8)
        self.fc = nn.Conv2d(conv_dim*8, num_classes+1, 4, 1, 0)              
        
    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.05)               # (?, 64, 32, 32)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.05)     # (?, 128, 16, 16)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.05)     # (?, 256, 8, 8)
        h = F.leaky_relu(self.bn4(self.conv4(h)), 0.05)     # (?, 512, 4, 4)
        out = self.fc(h)                                    # (?, 1, 1, 1)
        return out.squeeze()