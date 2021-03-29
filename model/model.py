"""
model by HankyuJang
inspired by dhiraj, fixa22

<CS492H: Machine Learning for 3D data>
Programming assignment 2

helped by 
https://www.kaggle.com/balraj98/modelnet40-princeton-3d-object-dataset/code
https://github.com/dhirajsuvarna/pointnet-autoencoder-pytorch/blob/master/train.py
https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetAE(nn.Module):
    def __init__(self, in_dim= 3, num_points= 2048):
        super(PointNetAE, self).__init__()
        
        self.in_dim = in_dim
        self.num_points = num_points

        self.encoder = PointEncoder(in_dim, num_points)
        self.decoder = PointDecoder(num_points)

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    

class PointEncoder(nn.Module):
    def __init__(self, in_dim = 3 ,  num_points = 2048):
        super(PointEncoder, self).__init__()

        self.in_dim = in_dim
        self.num_points = num_points
        
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        """
        Todo: check the accuracy and change the process making dim smaller.
        """
        self.fc1 = nn.Linear(1024, 512) # make global feature dimension smaller
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128) #latent space

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        """
        Input: (batch_size, n_points, in_dim).
        Outputs: 
        """
        batch_size = x.size()[0]
        num_points = x.size()[1]
        in_dim = x.size()[2]

        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = torch.max(x, 2, keepdim= True)[0]
        x = x.view(-1, 1024) # 1*1024

        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.fc3(x)

        return x


class PointDecoder(nn.Module):
    def __init__(self, num_points):
        super(PointDecoder, self).__init__()

        self.num_points = num_points
        
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points*3)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        x = x.view(-1, 3, self.num_points)

        return x


if __name__ == "__main__":
    print("This PointNet AutoEncoder constructed by Hankyu")

        