# Some code was borrowed from https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb 

from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

train_dataset = datasets.MNIST(
                    root = './mnist_data/',
                    train = True,
                    transform=transforms.ToTensor(),
                    download = True
                )

test_dataset = datasets.MNIST(
                    root = './mnist_data/',
                    train = False,
                    transform=transforms.ToTensor(),
                    download = False
                )

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = 128, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = 128, shuffle = False)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')


def experiment(epochs):

    model = VAE(img_dim = 784, z_dim = 20, hidden_dim = 512, dropout = 0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 2e-04)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for _, (x, _) in enumerate(train_loader):
            x = x.to(device)

            x_target, mu, sigma = model(x.view(-1,784))

            loss = vae_loss(x_target, x.view(-1,784), mu, sigma)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for _, (x_test, _) in enumerate(test_loader):
                x_test = x_test.to(device)

                x_target_test , mu_test, sigma_test = model(x_test.view(-1,784))

                loss_test = vae_loss(x_target_test, x_test.view(-1,784), mu_test, sigma_test)
                test_loss += loss_test.item()

            test_loss = test_loss / len(test_loader)

        print('===== epoch:{}. Training Loss: {:.3f} / Test Loss : {:.3f}. '
                                                .format(epoch, train_loss, test_loss))

    path = os.path.join('model_pt', 'vae.pt')
    torch.save(model.state_dict(), path)
    print("Trained model saved")    


def generate_img():
    
    model = VAE(img_dim = 784, z_dim = 20, hidden_dim = 512, dropout = 0.5).to(device)
    state_dict = torch.load(os.path.join('model_pt', 'vae.pt'), map_location = device)
    model.load_state_dict(state_dict, strict = False)
    
    model.eval()
    with torch.no_grad():
        latent_z = torch.randn(64,20).to(device)
        sample = model.decoder(latent_z).to(device)

        save_image(sample.view(64, 1, 28, 28), './samples/sample_2.png')



if __name__ == '__main__' :

    experiment(epochs = 40)
    generate_img()
    


