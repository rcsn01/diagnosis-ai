import torch
from PIL import Image
from torchvision import datasets, models, transforms,utils
import torch.nn as nn
import numpy as np
import random
import os
import torchvision
from tqdm import tqdm

#Set random seed
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
   os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(20)

root = './'

# Hyper parameters
num_epochs = 5    #number of cycles
batch_size = 2 #Amount of feeding data per time
learning_rate = 0.00005   #learning rate
momentum = 0.9  #rate of change
num_classes = 4 #number of classes


class MyDataset(torch.utils.data.Dataset):  # Create your own class: MyDataset, this class is inherited from torch.utils.data.Dataset
    def __init__(self, datatxt, transform=None, target_transform=None):  # Initialize some parameters that need to be put in
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  # According to the incoming path and txt text parameters, open this text and read the content
        imgs = []  # Create an empty list named img, which will be used to hold things later
        for line in fh:  # Loop content in txt text line by line
            line = line.rstrip()  # Delete the specified characters at the end of the string in this line. For a detailed introduction to this method, query python yourself.
            words = line.split()  # Slice the string by specifying the delimiter, which defaults to all null characters, including spaces, newlines, tabs, etc.
            imgs.append((words[0], int(words[1])))  # Read the content in the txt into the imgs list and save it. The specific number of words depends on the content of the txt.
            # Obviously, words[0] is picture information, words[1] is label
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # This method is required to read the specific content of each element according to the index.
        fn, label = self.imgs[index]  # fn is the image path #fn and label respectively obtain imgs[index], which is the information of word[0] and word[1] in each line just now
        img = Image.open(fn).convert('RGB')  # Read the image according to the path from PIL import Image # Read the image according to the path. The color image is RGB.
        img = img.resize((224,224))

        if self.transform is not None:
            img = self.transform(img)  # Whether to transform
        return img, label  # Return is very important. What content is returned, then what content can be obtained when we read each batch in a loop during training

    def __len__(self):  # This function must also be written. What it returns is the length of the data set, that is, the number of pictures. It must be distinguished from the length of the loader.
        return len(self.imgs)

# Create a data set based on the class MyDataset you defined! Note that it is a data set! instead of loader iterator
train_data = MyDataset(datatxt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(datatxt=root + 'test.txt', transform=transforms.ToTensor())

#Then call DataLoader and the data set just created to create the dataloader. Here, the length of the loader is how many batches there are, so it is related to batch_size.
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle=False)

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#Turn on gpu acceleration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MobileNet(nn.Module):
    def __init__(self, num_classes=num_classes):   # num_classes
        super(MobileNet, self).__init__()
        net = models.mobilenet_v2(pretrained=True)   # Load mobilenet_v2 network parameters from pre-trained model
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(    # Define your own classification layer
                nn.Linear(1280, 1000),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

net = MobileNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum = momentum )
optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,betas=(0.9,0.999))
# train_accs = []
# train_loss = []
test_acc2 = []
test_loss2 = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loader = tqdm(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward passx
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    net.eval()
    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = net(batch_x)
            loss2 = criterion(out, batch_y)
            test_loss += loss2.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            test_acc += num_correct.item()
        test_acc2.append(test_acc/len(test_data))
        test_loss2.append(test_loss/len(test_data))
        print('Epoch :{}, Test Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, test_loss / (len(
            test_data)), test_acc / (len(test_data))))

epochs = range(1, len(test_acc2)+1)
torch.save(net, 'model.ckpt')
print(test_acc2)
