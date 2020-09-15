import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.drop = nn.Dropout(p = dropout)

        self.linear1 = nn.Linear(
                            self.input_dim,
                            self.hidden_dim
                         )
        self.linear2 = nn.Linear(
                            self.hidden_dim,
                            self.hidden_dim,
                        )
        # In order to make mean, stddev, we take output dimension equals 2 * self.output_dim 
        self.out_layer = nn.Linear(
                            self.hidden_dim,
                            self.output_dim * 2
                        )

        # He Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
        
    def forward(self, x):

        x = F.elu(self.linear1(x))
        x = self.drop(x)

        x = torch.tanh(self.linear2(x))
        x = self.drop(x)

        params = self.out_layer(x)

        # mu : mean, sigma : standard deviation
        mu = params[:, :self.output_dim]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        sigma = 1e-6 + F.softplus(params[:, self.output_dim:])

        return mu, sigma


class Decoder(nn.Module):

    def __init__(self, z_dim, hidden_dim, output_dim, dropout = 0.5):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.drop = nn.Dropout(p = dropout)

        self.linear1 = nn.Linear(
                            self.z_dim,
                            self.hidden_dim
                         )
        self.linear2 = nn.Linear(
                            self.hidden_dim,
                            self.hidden_dim,
                        )
        self.linear3 = nn.Linear(
                            self.hidden_dim,
                            self.output_dim
                        )

        # He Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):

        x = torch.tanh(self.linear1(x))
        x = self.drop(x)

        x = F.elu(self.linear2(x))
        x = self.drop(x)

        x = torch.sigmoid(self.linear3(x))

        return x 


# In order to use backpropagation, reparametrize sampling 
class VAE(nn.Module):
    
    def __init__(self, x_hat, x_target, img_dim, z_dim, hidden_dim, dropout):
        super().__init__()

        self.x_hat = x_hat
        self.x_target = x_target
        
        self.img_dim = img_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        
        self.dropout = dropout

    def loss(self):

        batch_size = self.x_hat.size(0)
        
        # sampling by re-parametrization in order to use back-propagation
        encoder = Encoder(self.img_dim, self.hidden_dim, self.img_dim, self.dropout)
        mu, sigma = encoder(self.x_hat)
        z = mu + sigma * torch.randn(mu.shape)

        decoder = Decoder(self.img_dim, self.hidden_dim, self.img_dim, self.dropout)
        y = decoder(z)
        y = torch.clamp(y, 1e-8, 1 - 1e-8)

        # Loss = reconstruction error + KL divergence loss
        generative_loss = F.binary_cross_entropy(y, self.x_target)

        KLD_loss = 0.5 * torch.sum(
                                    torch.pow(mu, 2) +
                                    torch.pow(sigma, 2) -
                                    torch.log(1e-8 + torch.pow(sigma, 2)) -1
                                ).sum() / batch_size
        
        loss = generative_loss + KLD_loss
        
        return loss

        
    def get_ae(self, x):

        encoder = Encoder(x.shape[1], self.hidden_dim, self.z_dim, self.dropout)
        mu, sigma = encoder(x)

        z = mu + sigma * torch.randn(mu.shape)

        decoder = Decoder(self.z_dim , self.hidden_dim, self.img_dim, 1.0)
        y = decoder(z)
        return y




    

# a = torch.rand(10, 30)
# b = torch.rand(10, 30)
# c = torch.rand(10, 30)

# model = VAE(a,b, 20, 40, 40, 0.5)
# model.eval()
# print(model.get_ae(c))







