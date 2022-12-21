# H. Su, S. Maji, E. Kalogerakis, and E. Learned-Miller, “Multi-view convolutional neural networks for 3D shape recognition,” in Proc. IEEE Int. Conf. Comput. Vis., Dec. 2015, pp. 945–953
import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class Model(nn.Module): # input 256*256 per 2 angolazioni
    def __init__(self):
        super(Model, self).__init__()

        self.cnn = torchvision.models.vgg11(pretrained=False, progress=True)#resnet18
        self.cnn.classifier = nn.Identity()#fc

        self.classifier = nn.Linear(512*7*7,2)#512
        
    def forward(self,inputs):
        imgs, doppiaAngolazione = inputs

        out1 = self.cnn(imgs)
        out2 = self.cnn(doppiaAngolazione)

        pooled_view = torch.max(out1, out2)

        out = self.classifier(pooled_view)

        return out
    
