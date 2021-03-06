def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

      
use_ipython = run_from_ipython()
      
if use_ipython:
    from   IPython import display
  
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torchvision import datasets, transforms


if use_ipython:
    plt.ioff
else:
    plt.ion()


if os.path.isdir("results"):
    shutil.rmtree('results')
if not os.path.isdir("results"):
    os.mkdir("results")


use_cuda = True if torch.cuda.is_available() else False


if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


# Normalize Images
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

batch_size = 100
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
   

image_size = 28*28


def SingleLayer(in_feature, out_feature):
    return (
        nn.Linear(in_feature, out_feature),
#       nn.LeakyReLU(0.2, inplace=True),
        nn.ReLU(),
        nn.Dropout(0.3),    
    )

def add(list, *argv):
    for arg in argv:
        list.append(arg)


class Discriminator(nn.Module):
  
    def __init__(self, image_size):
        super().__init__()
        self.model = nn.Sequential(
            *SingleLayer(image_size, 1024),
            *SingleLayer(1024, 512),
            *SingleLayer(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.model(x.view(x.size(0), 784))
        out = out.view(out.size(0), -1)
        return out


class Generator(nn.Module):
  
    def __init__(self, image_size):
        super().__init__()
        self.model = nn.Sequential(
            *SingleLayer(100, 256),
            *SingleLayer(256, 512),
            *SingleLayer(512, 1024),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )
    
    
    def forward(self, x):
        x = x.view(x.size(0), 100)
        out = self.model(x)
        return out


discriminator = Discriminator(image_size).cuda()
generator = Generator(image_size).cuda()


criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)


def train_discriminator(discriminator, images, real_labels, fake_images, fake_labels):
    discriminator.zero_grad()

    outputs = discriminator(images)
    real_loss = criterion(outputs, real_labels)
    real_score = outputs
    
    outputs = discriminator(fake_images) 
    fake_loss = criterion(outputs, fake_labels)
    fake_score = outputs

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss, real_score, fake_score


def train_generator(generator, discriminator_outputs, real_labels):
    generator.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels)
    g_loss.backward()
    g_optimizer.step()
    return g_loss


num_test_samples = 16


# create figure for plotting
size_figure_grid = int(math.sqrt(num_test_samples))
fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    ax[i,j].get_xaxis().set_visible(False)
    ax[i,j].get_yaxis().set_visible(False)


# set number of epochs and initialize figure counter
num_epochs = 500
num_batches = len(train_loader)
num_fig = 0


for epoch in range(num_epochs):
    for n, (images, _) in enumerate(train_loader):
        images = Variable(images.cuda())
        real_labels = Variable(torch.ones(images.size(0)).cuda())
        
        # fake images from generator
        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = generator(noise)
        fake_labels = Variable(torch.zeros(images.size(0)).cuda())
        
        # train the discriminator with real images, fake images.
        d_loss, real_score, fake_score = train_discriminator(discriminator, images, real_labels, fake_images, fake_labels)
        
        # face images to train generator
        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = generator(noise)
        outputs = discriminator(fake_images)

        # train the generator
        g_loss = train_generator(generator, outputs, real_labels)
        
        if (n+1) % 200 == 0:
            test_noise = Variable(torch.randn(num_test_samples, 100).cuda())
            test_images = generator(test_noise)
            
            for k in range(num_test_samples):
                i = k//4
                j = k%4
                ax[i,j].cla()
                ax[i,j].imshow(test_images[k,:].data.cpu().numpy().reshape(28, 28), cmap='Greys')
            
            if use_ipython:
              display.clear_output(wait=True)
              display.display(plt.gcf())
            else:
              plt.show()
            
            plt.savefig('results/mnist-gan-%03d.png'%num_fig)
            num_fig += 1
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
                  'D(x): %.2f, D(G(z)): %.2f' 
                  %(epoch + 1, num_epochs, n+1, num_batches, d_loss.data[0], g_loss.data[0],
                    real_score.data.mean(), fake_score.data.mean()))

fig.close()

