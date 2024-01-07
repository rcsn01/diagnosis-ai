import warnings
warnings.filterwarnings("ignore")
import torch
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms,models
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


classes = ['BacterialPneumoniaXray','Covid19','NormalXray','ViralPneumoniaXray']  #Tag serial number corresponds to class name
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  #Whether to use gpu acceleration
num_classes = 4

class MobileNet(nn.Module):
    def __init__(self, num_classes=num_classes):
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

def test_mydata(a):   #Define prediction function
    # Resize image
    im = plt.imread(a)     #Read in pictures
    images = Image.open(a)     #Store pictures in images
    images = images.resize((224, 224))   #Resize the image to 224*224
    images = images.convert('RGB')    #RGB

    transform = transforms.ToTensor()
    images = transform(images)   #Convert image to tensor type
    images = images.resize(1, 3, 224, 224)    #Adjust the size of the input network image
    images = images.to(device)   #gpu acceleration

    path_model = "./model.ckpt"   #Call the trained model
    model = torch.load(path_model)
    model = model.to(device)

    model.eval()
    outputs = model(images)    #Pass the image into the model for prediction
    values, indices = outputs.data.max(1)  #Returns the maximum probability value and subscript. The output is not of tensor type, so .data must be added.
    print(classes[int(indices[0])])   #Output class name
    plt.title(classes[int(indices[0])])   
    plt.imshow(im)    #show results
    plt.show()


while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = test_mydata(img)



