import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_data():
    transform_train = transforms.Compose([
        transforms.Resize((28, 28)), # different size data
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(root='./dataset/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)
    testset = torchvision.datasets.ImageFolder(root='./dataset/', transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

    classes = ['corbo', 'not_corbo']
    return {'train': trainloader, 'test': testloader, 'classes': classes}

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=0, bias=False)
        self.conv2 = nn.Conv2d(6, 12, 5, padding=0, bias=False)
        self.fc1 = nn.Linear(192, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # print(x.shape)
        x = torch.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        return x

def train(net, dataloader, epochs=1, lr=0.01, momentum=0.9, decay=0.0, verbose=1):
#   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  device = torch.device("cpu")
  net.to(device)
  losses = []
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
  for epoch in range(epochs):
    sum_loss = 0.0
    for i, batch in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch[0].to(device), batch[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize 
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        losses.append(loss.item())
        sum_loss += loss.item()
        if i % 5 == 1:
            if verbose:
              print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, sum_loss / 100))
            sum_loss = 0.0
  return losses

def accuracy(net, dataloader):
  correct = 0
  total = 0
  device = torch.device("cpu")
  with torch.no_grad():
      for batch in dataloader:
          images, labels = batch[0].to(device), batch[1].to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  return correct/total

def smooth(x, size):
  return np.convolve(x, np.ones(size)/size, mode='valid')

def main():
    data = get_data()
    print(data['train'].__dict__)
    conv_net = ConvNet()
    conv_losses = train(conv_net, data['train'], epochs=3)
    torch.save(conv_net.state_dict(), './model')
    plt.plot(smooth(conv_losses, 50))

    print("Training accuracy: {}".format(accuracy(conv_net, data['train'])))
    print("Testing accuracy: {}".format(accuracy(conv_net, data['test'])))

if __name__ == "__main__":
   main()