#
# orginal source codes from: https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py
#

import numpy as np
import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if True:
    def raw_preprocess(data):
        return data

    def input_func(x):
        return x

    print("Raw data")

    preprocess = raw_preprocess
    d_input_func = input_func
else:
    def decorate_with_diffs(data):
        exponent = 2.0
        mean = torch.mean(data.data, 1, keepdim=True)
        mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
        diffs = torch.pow(data - Variable(mean_broadcast), exponent)
        return torch.cat([data, diffs], 1)
        
    def input_func(x):
        return x * 2

    print("Data and variances")

    preprocess = decorate_with_diffs
    d_input_func = input_func

#
# Models
#

#
# Generator model
#

class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):

        x = self.linear1(x)
        x = F.elu(x)

        x = self.linear2(x)
        x = F.sigmoid(x)

        return self.linear3(x)

#
# Discriminator model
#

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.linear1(x)
        x = F.elu(x)

        x = self.linear2(x)
        x = F.elu(x)

        x = self.linear3(x)
        return F.sigmoid(x)


#
# Sampler
#

def get_normal_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))

def get_uniform_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)


d_sampler = get_normal_distribution_sampler(4, 1.25)
gi_sampler = get_uniform_generator_input_sampler()

#
# Network specifications
#

g_input_size = 1
g_hidden_size = 50
g_output_size = 1

d_input_size = 100
d_hidden_size = 50
d_output_size = 1

minibatch_size = d_input_size
    
#
# Adversarial
#

G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
criterion = nn.BCELoss()

#
# Optimizer
#

optim_betas = (0.9, 0.999)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=optim_betas)


for epoch in range(100000):

    k_steps = 1
    for d_index in range(k_steps):

        D.zero_grad()

        #  Train Discriminator with REAL data
        d_sample = d_sampler(d_input_size) # size:[1, 100]
        if epoch == 0:
            print("d_sample")
            print(d_sample.size())

        d_real_data = Variable(d_sample)
        d_real_output = D(preprocess(d_real_data))
        d_real_error = criterion(d_real_output, Variable(torch.ones(1)))
        d_real_error.backward()

        if epoch == 0:
            print("d_real_error")
            print(d_real_error.size())

        # Train Discriminator with FAKE data
        gi_sample = gi_sampler(minibatch_size, g_input_size) # size:[100, 1]
        if epoch == 0:
            print("gi_sample")
            print(gi_sample.size())

        d_input = Variable(gi_sample)
        d_fake_output = G(d_input)
        d_fake_data = d_fake_output.detach()
        d_fake_output = D(preprocess(d_fake_data.t()))
        d_fake_error = criterion(d_fake_output, Variable(torch.zeros(1)))
        d_fake_error.backward()
        d_optimizer.step()

        if epoch == 0:
            print("d_fake_error")
            print(d_fake_error.size())

    g_steps = 1
    for g_index in range(g_steps):

        G.zero_grad()

        gi_sample = gi_sampler(minibatch_size, g_input_size) # size:[100, 1]
        if epoch == 0:
            print("gi_sample")
            print(gi_sample.size())

        g_input = Variable(gi_sample)
        g_fake_data = G(g_input)
        g_fake_output = D(preprocess(g_fake_data.t()))
        g_error = criterion(g_fake_output, Variable(torch.ones(1)))
        g_error.backward()
        g_optimizer.step()

        if epoch == 0:
            print("g_error")
            print(g_error.size())

    if epoch % 200 == 0:
        print("%5s: D: %2.2f/%2.2f G: %2.2f (Real: %2.2f, %2.2f) Fake: (%2.2f, %2.2f) " % (
            epoch,
            d_real_error.data.storage().tolist()[0],
            d_fake_error.data.storage().tolist()[0],
            g_error.data.storage().tolist()[0],
            np.mean(d_real_data.data.storage().tolist()), np.std(d_real_data.data.storage().tolist()),
            np.mean(d_fake_data.data.storage().tolist()), np.std(d_fake_data.data.storage().tolist())))

