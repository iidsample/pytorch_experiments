import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

training_param_dict = {'epochs':10, 'batch_size':64, 'test_batch_size':100,
                       'learning_rate' : 0.01, 'momentum':0.5, 'cuda':True,
                       'random_seed':1, 'log_interval' :1000}

if not torch.cuda.is_available():
    training_param_dict['cuda'] = False

torch.manual_seed(training_param_dict['random_seed'])
if training_param_dict['cuda']:
    torch.cuda.manual_seed(training_param_dict['random_seed'])
    kwargs = {'num_workers' : 1, 'pin_memory' : True}

else:
    kwargs = dict()

# will look at the data loading part a little later
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=training_param_dict['batch_size'], shuffle=True, **kwargs)
# import ipdb; ipdb.set_trace()
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=training_param_dict['batch_size'], shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if training_param_dict['cuda']:
    model.cuda()

optimizer = optim.SGD(model.parameters(), 
                      lr=training_param_dict['learning_rate'],
                      momentum=training_param_dict['momentum'])

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if training_param_dict['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % training_param_dict['log_interval']:
            print ('Train epoch : {} [ {} / {} ({:.0f} %)] \tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                           len(train_loader.dataset),
                           100.*batch_idx/len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if training_param_dict['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    test_loss = test_loss
    test_loss /= len(test_loader)
    print('\nTest set: Average loass: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
                              100. * correct / len(test_loader.dataset)))

for epoch in range(1, training_param_dict['epochs'] + 1):
    train(epoch)
    test(epoch) 
