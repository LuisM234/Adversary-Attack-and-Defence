import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time
import copy



# =============================================================================
# D.2.2 Define Competition ID
# =============================================================================
# Redacted my student ID for privacy reasons.
# id_ = 

# =============================================================================
# D.2.3 Set Training Parameters
# =============================================================================
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args([])

# =============================================================================
# D.2.4 Toggle GPU/CPU
# =============================================================================
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# =============================================================================
# D.2.5 Loading Dataset and Define Network Structure
# =============================================================================
train_set = torchvision.datasets.FashionMNIST(root='data', train=True,
                                               download=True, 
                                               transform=transforms.Compose([
                                                   transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='data', train=False,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

# Define fully connected network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

# =============================================================================
# D.2.6 Adversarial Attack - Implementation
# =============================================================================
def adv_attack(model, X, y, device):
    """
    Projected Gradient Descent (PGD) Attack
    A stronger iterative version of FGSM
    """
    X_adv = Variable(X.data)
    
    #  parameters
    epsilon = 0.2 # Maximum perturbation
    alpha = 0.01   # Step size
    num_iter = 40  # Number of iterations
    
    for i in range(num_iter):
        X_adv_var = Variable(X_adv, requires_grad=True)
        
        # Forward pass
        output = model(X_adv_var)
        loss = F.nll_loss(output, y)
        
        # Backward pass
        loss.backward()
        
        # Get gradient sign
        grad = X_adv_var.grad.data
        
        # Update adversarial example
        X_adv = X_adv + alpha * grad.sign()
        
        # Project back to epsilon ball
        eta = torch.clamp(X_adv - X.data, -epsilon, epsilon)
        X_adv = torch.clamp(X.data + eta, 0, 1)
    
    return X_adv

def adv_attack_train(model, X, y, device, epsilon=0.1, num_iter=10):

    X_adv = Variable(X.data)
    alpha = 0.01
    
    for i in range(num_iter):
        X_adv_var = Variable(X_adv, requires_grad=True)
        output = model(X_adv_var)
        loss = F.nll_loss(output, y)
        loss.backward()
        grad = X_adv_var.grad.data
        X_adv = X_adv + alpha * grad.sign()
        eta = torch.clamp(X_adv - X.data, -epsilon, epsilon)
        X_adv = torch.clamp(X.data + eta, 0, 1)
    
    return X_adv

# =============================================================================
# D.2.7 Evaluation Functions
# =============================================================================
def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28*28)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_adv_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28*28)
        adv_data = adv_attack(model, data, target, device=device)
        
        
        with torch.no_grad():
            output = model(adv_data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# =============================================================================
# D.2.8 Adversarial Training
# =============================================================================
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28*28)
        
        optimizer.zero_grad()
        
        # Mix clean and progressively stronger adversarial training
        if batch_idx % 5 == 0:
            
            output = model(data)
        else:
           
            # Gradually increase epsilon as training progresses
            epsilon_train = 0.20  
            adv_data = adv_attack_train(model, data, target, device, epsilon=epsilon_train, num_iter=40)
            output = model(adv_data)
        
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


# =============================================================================
# Main Training Loop
# =============================================================================
def train_model():
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Training
        train(args, model, device, train_loader, optimizer, epoch)
        
        # Evaluation
        trnloss, trnacc = eval_test(model, device, train_loader)
        advloss, advacc = eval_adv_test(model, device, test_loader)
        
        print(f'Epoch {epoch}: {int(time.time() - start_time)}s')
        print(f'trn_loss: {trnloss:.4f}, trn_acc: {trnacc:.2f}')
        print(f'adv_loss: {advloss:.4f}, adv_acc: {advacc:.2f}')
        
        adv_tatloss, adv_tatacc = eval_adv_test(model, device, test_loader)
        print(f'Your estimated attack ability: {adv_tatacc:.4f}')
        print(f'Your estimated defence ability: {adv_tatacc:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), str(id_) + '.pt')
    return model

# =============================================================================
# D.2.9 Define Distance Metrics
# =============================================================================
def p_distance(model, train_loader, device):
    p = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28*28)
        adv_data = adv_attack(model, data, target, device=device)
        p.append(torch.norm(data - adv_data, float('inf')))
    print('epsilon p:', max(p))

# =============================================================================
# D.2.10 Supplementary Code for Test Purpose
# =============================================================================
# Comment out when submitting
#if __name__ == "__main__":
 #   model = train_model()
 #  p_distance(model, train_loader, device)
