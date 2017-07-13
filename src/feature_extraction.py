import time

import torch
from torchvision import models
from torch import nn
import torchvision.transforms as transforms
from torch.autograd import Variable

import PIL
from PIL import Image

orignal_models = models.alexnet(pretrained=True)

class AlexnetConv4(nn.Module):
    def __init__(self):
        super(AlexnetConv4, self).__init__()
        self.features = nn.Sequential(
            *list(orignal_models.features.children())[:-3]
        )
    def forward(self, x):
        x = self.features(x)
        return (x)

model = AlexnetConv4()
print (model)

image_transforms = transforms.Compose(
                        [transforms.Scale((256,256), PIL.Image.NEAREST),transforms.CenterCrop(224),
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485,
                                                                         0.456,
                                                                         0.406],
                                                                   std=[0.229,
                                                                        0.224,
                                                                        0.225])])
# model.cuda()
model.eval()

img_file = Image.open('/home/saurabh/Downloads/Latest-Bags-Trends.jpg')
img_file_trans = image_transforms(img_file)
img_file_trans = img_file_trans.unsqueeze(0)

print (img_file_trans)
start_time = time.time()
img_file_trans = Variable(img_file_trans, volatile=True)
end_time = time.time()
print('time taken {} ms'.format((end_time - start_time)*1000))
output = model(img_file_trans)
print (output)

