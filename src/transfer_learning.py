import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable

pretrained_model = models.alexnet(pretrained=True)
image_transforms = transforms.Compose(
    [transforms.Scale((256,256)), transforms.CenterCrop(224),
     transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],
                                                                   std=[0.229,
                                                                        0.224,
                                                                        0.225])])

class Snaper(nn.Module):
    def __init__(self):
        super(Snaper, self).__init__()
        self.features = nn.Sequential(*list(
            pretrained_model.features.children())[:-1])
        self.snap_on = nn.Sequential(
            nn.Linear(43264, 4500),
            nn.ReLU(inplace=True), 
            nn.Linear(4500, 2250), 
            nn.ReLU(inplace=True),
            nn.Linear(2250,51)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 43264)
        x = self.snap_on(x)
        return (x)

# model = Snaper()
# model.cuda()

def train(model, epoch, train_loader, optimizer):
    """
    runs over each epoch over train loader object
    """
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        print( 'calculating the loss')
        loss = F.nll_loss(output, target)
        print ('started loss backward')
        loss.backward()
        print ('started optimizer step')
        optimizer.step()
        if batch_index % 100:
            print ('Train epoch : {} [ {}/{} ({:.0f} %)] \t Loss : {:.6f}'.format(
                epoch, batch_index*len(data), len(train_loader.dataset), 100. *
                batch_index/len(train_loader), loss.data[0]))


def get_class(dir_path):
    classes = [d for d in os.listdir(dir_path) if
               os.path.isdir(os.path.join(dir_path, d))]
    classes.sort()
    print (classes)


def test(model, epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    test_loss = test_loss
    test_loss /= len(test_loader)
    print ("""Test Statics \n AverageLoss : {:.4f} \n Accuracy : {}/{}
    ({:.1f})""".format(test_loss, correct, len(test_loader.dataset),
                       100.*correct / len(test_loader.dataset)))

def main():
    model = Snaper()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), 
                      lr=0.01,
                      momentum=0.5)

    print ('training loader reading data')
    train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
    '/media/saurabh/Extra_1/training_data/macy_training/train', image_transforms),
                                           num_workers=12, batch_size= 64)
    print ('started var loader ')
    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
        '/media/saurabh/Extra_1/training_data/macy_training/valid',
        image_transforms), num_workers =12, batch_size=64)
    print ('read data')
    for epoch in range(1,100):
        train(model, epoch, train_loader, optimizer)
        test(model, epoch, val_loader)
if __name__ == '__main__':
    main()
