from model import Generator
from torch.autograd import Variable
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image, make_grid

g = Generator()
g.eval()
g.cuda()
g.load_state_dict(torch.load('./250000-G.pkl'))


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    print 'low', low
    print 'high',high
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    print 'val', val
    print 'so', so
    print 'omega', omega
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def interpolate(z1, z2, z3, z4, n=6):
    z1 = np.asarray(z1).astype(np.float32)       # 1D vector of shape [100].
    z2 = np.asarray(z2).astype(np.float32)
    z3 = np.asarray(z3).astype(np.float32)
    z3 = np.asarray(z4).astype(np.float32)
    print type(z3)
    print z3.shape
    print z3
    print 
    z_left = []
    z_right = []
    for ratio in np.linspace(0, 1, n):
        z_left.append(slerp(ratio, z1, z2)) 
        print "ratio", ratio
        z_right.append(slerp(ratio, z3, z4))
        print "done========================================"
        print ratio
    zs = []
    for z_l, z_r in zip(z_left, z_right):
        for ratio in np.linspace(0, 1, n):
            zs.append(slerp(ratio, z_l, z_r))

    z = torch.from_numpy(np.asarray(zs))
    z = z.view(z.size(0), 100, 1, 1)
    samples = g(Variable(z.cuda()))

    grid = make_grid(samples.data, nrow=6, padding=0)
    grid = (grid[:, :-1, :-1] * 255).long()                              # (36, 3, 384, 384)
    grid = grid.cpu().numpy().transpose(1, 2, 0).reshape(-1).tolist()
    return grid