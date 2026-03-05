
from tracemalloc import start
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th
from torch.nn.modules.linear import Linear

import torchvision.models as pre_models
import numpy as np
import torch.nn.functional as F

'''
Here we provide 5 feature extractor networks

1. No_CNN
    No CNN layers
    Only maxpooling layer to generate 25 features

2. CNN_GAP
    3 layers of CNN
    finished by AvgPool2d
    1*8 -> 8*16 -> 16*25

3. CNN_GAP_BN
    3 layers of CNN with BN for each CNN layer
    finished by AvgPool2d

4. CNN_FC
    3 layers of CNN
    finished by Flatten
    FC is used to get CNN features (960 100 25)

5. CNN_MobileNet
    Using a pre-trained MobileNet as feature generator
    finished by Flatten (576 -> 25)
'''


class No_CNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=4):
        super(No_CNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # Can use model.actor.features_extractor.feature_all to print all features

        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_all = None

        # input size 80*100
        # divided by 5
        self.cnn = nn.Sequential(
            nn.MaxPool2d(kernel_size=(16, 20)),
            # nn.MaxPool2d(kernel_size=(26, 33)),
            nn.Flatten()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        cnn_feature = self.cnn(depth_img)  # [1, 25, 1, 1]
        # print(cnn_feature)
        # print(self.feature_num_state)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1
        # print(state_feature.size(), cnn_feature.size())
        x = th.cat((cnn_feature, state_feature), dim=1)
        # print(x)
        self.feature_all = x  # use  to update feature before FC

        return x
 

class No_CNN_Dual(BaseFeaturesExtractor):
    """
    Dual-stream policy: Extracts both 'Max' (nearest obstacle) and 
    'Avg' (obstacle density) features from the depth image.
    
    Designed to provide richer gradients than standard No_CNN for faster convergence.
    Input:  [Batch, 1, 80, 100] (Depth Image)
    Output: [Batch, 50 + state_dim] (Flattened 25 Max + 25 Avg features + state)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=4):
        # features_dim is a placeholder; we calculate the real dim dynamically
        super(No_CNN_Dual, self).__init__(observation_space, features_dim)

        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_all = None

        # 1. Danger Stream: MaxPool (Captures nearest obstacle in the grid)
        # Input: 80x100 -> Output: 5x5
        self.max_pool = nn.MaxPool2d(kernel_size=(16, 20))

        # 2. Density Stream: AvgPool (Captures average clutter in the grid)
        # Input: 80x100 -> Output: 5x5
        self.avg_pool = nn.AvgPool2d(kernel_size=(16, 20))

        # 3. Flatten layer for both streams
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass to ensure compatibility
        # We manually normalize by 255.0 here to mimic SB3's preprocessing during init
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None][:, 0:1, :, :]).float()
            dim_max = self.flatten(self.max_pool(sample_input)).shape[1]
            dim_avg = self.flatten(self.avg_pool(sample_input)).shape[1]
            
        # Update the features_dim to match actual output + state features
        self._features_dim = dim_max + dim_avg + state_feature_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Note: observations are already normalized to [0, 1] by SB3
        depth_img = observations[:, 0:1, :, :]
        
        # Stream 1: Max Features (25 features) - High value means close obstacle
        feat_max = self.flatten(self.max_pool(depth_img))
        
        # Stream 2: Avg Features (25 features) - High value means high density
        feat_avg = self.flatten(self.avg_pool(depth_img))
        
        # Concatenate CNN features [Batch, 50]
        cnn_feature = th.cat((feat_max, feat_avg), dim=1)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]

        # Final Concatenate
        out = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = out

        return out


class CNN_GAP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_GAP, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [1, 8, 40, 48]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 20, 24]
            # nn.BatchNorm2d(8, affine=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, self.feature_num_cnn, kernel_size=3,
                      stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 10, 12]
        )
        self.gap_layer = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

        self.batch_layer = nn.BatchNorm1d(self.feature_num_cnn)

        # nn.init.kaiming_normal_(self.conv1[0].weight, a=0, mode='fan_in')
        # nn.init.kaiming_normal_(self.conv2[0].weight, a=0, mode='fan_in')
        # nn.init.kaiming_normal_(self.conv3[0].weight, a=0, mode='fan_in')
        # nn.init.constant(self.conv1[0].bias, 0.0)
        # nn.init.constant(self.conv2[0].bias, 0.0)
        # nn.init.constant(self.conv3[0].bias, 0.0)

        # nn.init.xavier_uniform(self.conv1[0].weight)
        # nn.init.xavier_uniform(self.conv2[0].weight)
        # nn.init.xavier_uniform(self.conv3[0].weight)
        # self.conv1[0].bias.data.fill_(0)
        # self.conv2[0].bias.data.fill_(0)
        # self.conv3[0].bias.data.fill_(0)
        # self.soft_max_layer = nn.Softmax(dim=1)
        # self.batch_norm_layer = nn.BatchNorm1d(16, affine=False)

        # self.linear = self.cnn

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        self.layer_1_out = self.conv1(depth_img)
        self.layer_2_out = self.conv2(self.layer_1_out)
        self.layer_3_out = self.conv3(self.layer_2_out)
        self.gap_layer_out = self.gap_layer(self.layer_3_out)

        cnn_feature = self.gap_layer_out  # [1, 8, 1, 1]
        cnn_feature = cnn_feature.squeeze(dim=3)  # [1, 8, 1]
        cnn_feature = cnn_feature.squeeze(dim=2)  # [1, 8]
        # cnn_feature = th.clamp(cnn_feature,-1,2)
        # cnn_feature = self.batch_layer(cnn_feature)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC

        return x


class CNN_GAP_BN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_GAP_BN, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [1, 8, 40, 48]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 20, 24]
            # nn.BatchNorm2d(8, affine=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, self.feature_num_cnn, kernel_size=3,
                      stride=1, padding='same'),
            nn.BatchNorm2d(self.feature_num_cnn),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 10, 12]
        )
        self.gap_layer = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

        self.batch_layer = nn.BatchNorm1d(self.feature_num_cnn)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        self.layer_1_out = self.conv1(depth_img)
        self.layer_2_out = self.conv2(self.layer_1_out)
        self.layer_3_out = self.conv3(self.layer_2_out)
        self.gap_layer_out = self.gap_layer(self.layer_3_out)

        cnn_feature = self.gap_layer_out  # [1, 8, 1, 1]
        cnn_feature = cnn_feature.squeeze(dim=3)  # [1, 8, 1]
        cnn_feature = cnn_feature.squeeze(dim=2)  # [1, 8]
        # cnn_feature = th.clamp(cnn_feature,-1,2)
        # cnn_feature = self.batch_layer(cnn_feature)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC

        return x


class CustomNoCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=4):
        super(CustomNoCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # Can use model.actor.features_extractor.feature_all to print all features

        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_all = None

        # input size 80*100
        # divided by 5
        self.cnn = nn.Sequential(
            nn.MaxPool2d(kernel_size=(16, 20)),
            nn.Flatten()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        cnn_feature = self.cnn(depth_img)  # [1, 25, 1, 1]

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1
        # print(state_feature.size(), cnn_feature.size())
        x = th.cat((cnn_feature, state_feature), dim=1)
        # print(x)
        self.feature_all = x  # use  to update feature before FC

        return x


class CNN_FC(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_FC, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        # Input image: 80*100
        # Output: 16 CNN features + n state features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 40, 48]

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 20, 24]

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 10, 12]

            # nn.BatchNorm2d(8),
            nn.Flatten(),   # 960
            # nn.AvgPool2d(kernel_size=(10, 12), stride=1)
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[
                             None][:, 0:1, :, :]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 100),
            nn.ReLU(),
            nn.Linear(100, self.feature_num_cnn),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        cnn_feature = self.linear(self.cnn(depth_img))
        # cnn_feature = cnn_feature.squeeze(dim=3) # [1, 8, 1]
        # cnn_feature = cnn_feature.squeeze(dim=2) # [1, 8]

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC
        # print(x)

        return x


class CNN_MobileNet(BaseFeaturesExtractor):
    '''
    Using part of mobile_net_v3_small to generate features from depth image
    '''

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_MobileNet, self).__init__(observation_space, features_dim)

        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.mobilenet_v3_small = pre_models.mobilenet_v3_small(
            pretrained=True)

        self.part = self.mobilenet_v3_small.features

        # freeze part parameters
        for param in self.part.parameters():
            param.requires_grad = False

        self.gap_layer = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Sequential(
            nn.Linear(576, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(256, self.feature_num_cnn),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(0.25)
        )
        self.linear_small = nn.Sequential(
            nn.Linear(576, self.feature_num_cnn),
            nn.Tanh(),
            # nn.BatchNorm1d(32),
            # nn.Dropout(0.25)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        # change input image to (None, 3, 100, 80)
        # notion: this repeat is used for tensor  # (1, 3, 80 ,100)
        depth_img_stack = depth_img.repeat(1, 3, 1, 1)

        self.last_cnn_output = self.part(
            depth_img_stack)        # [1, 576, 3, 4]
        self.gap_layer_out = cnn_feature = self.gap_layer(
            self.last_cnn_output)  # [1, 576, 1, 1]

        cnn_feature = cnn_feature.squeeze(dim=3)  # [1, 576, 1]
        cnn_feature = cnn_feature.squeeze(dim=2)  # [1, 576]
        cnn_feature = self.linear_small(cnn_feature)  # [1, 32]

        state_feature = observations[:, 1, 0,
                                     0:self.feature_num_state]  # [1, 2]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC
        # print(x)

        return x


class CNN_GAP_new(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_GAP_new, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        # input size (100, 80)
        # input size 80 60
        self.conv1 = nn.Conv2d(1, 8, 5, 2)  # 28,38
        self.conv2 = nn.Conv2d(8, 8, 5, 2)  # 12,17
        self.conv3 = nn.Conv2d(8, 8, 3, 2)  # 5, 8
        self.pool = nn.MaxPool2d(2, 3)
        self.gap_layer = nn.AvgPool2d(kernel_size=(8, 10), stride=1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]  # 0-1 0->20m 1->0m
        # print(th.min(depth_img), th.max(depth_img))
        # norm image to (-1, 1)
        depth_img_norm = (depth_img - 0.5) * 2
        # print(th.min(depth_img_norm), th.max(depth_img_norm))

        # 1, 8, 38, 48  1,8,28,38
        self.layer_1_out = F.relu(self.conv1(depth_img_norm))
        # 1, 8, 18, 23  1,8,12,17
        self.layer_2_out = F.relu(self.conv2(self.layer_1_out))
        self.layer_3_out = F.relu(self.conv3(
            self.layer_2_out))  # 1, 16, 8, 10  1,8,5,8
        self.layer_small = self.pool(self.layer_3_out)  # 1,8,2,3
        # self.gap_layer_out = self.gap_layer(self.layer_3_out)               # 1, 16, 1, 1
        self.flatten = th.flatten(self.layer_small, start_dim=1)
        # self.flatten = self.flatten.unsqueeze(0)

        # cnn_feature = self.gap_layer_out  # [1, 16, 1, 1]
        # cnn_feature = cnn_feature.squeeze(dim=3)  # [1, 16, 1]
        # cnn_feature = cnn_feature.squeeze(dim=2)  # [1, 16]
        # cnn_feature = th.clamp(cnn_feature, -1, 2)
        # cnn_feature = self.batch_layer(cnn_feature)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((self.flatten, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC

        return x


class CNN_Spatial(BaseFeaturesExtractor):
    """
    Hybrid policy: Combines the learnable filters of a CNN with the 
    spatial preservation of No_CNN.
    
    Designed specifically for 80x100 input to output a 5x5 feature grid.
    Input:  [Batch, 1, 80, 100] (Depth Image)
    Output: [Batch, feature_num_cnn + state_dim]
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=4):
        # features_dim is the total dimension (CNN latent + state)
        super(CNN_Spatial, self).__init__(observation_space, features_dim)

        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        # 1. Learnable Layer: Extract features but keep size roughly same
        # Input: 80x100
        # Output: 8 x 80 x 100 (Channels x H x W)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )

        # 2. Spatial Downsampling: Mimic No_CNN's grid
        # We want final grid to be roughly 5x5 to preserve spatial info.
        # 80 / 16 = 5
        # 100 / 20 = 5
        self.pool = nn.MaxPool2d(kernel_size=(16, 20))
        
        # 3. Flatten
        self.flatten = nn.Flatten()

        # Compute shape of flattened spatial features
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None][:, 0:1, :, :]).float()
            n_flatten = self.flatten(self.pool(self.conv(sample_input))).shape[1] # Should be 200
        
        # 4. Compression Layer: Reduce 200 features to feature_num_cnn (e.g., 32)
        # This prevents the vision features from "drowning out" the 3-4 state features
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, self.feature_num_cnn),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Note: observations are already normalized to [0, 1] by SB3
        depth_img = observations[:, 0:1, :, :]
        
        # Optional: Normalize to [-1, 1] for better gradient flow
        depth_img_norm = (depth_img - 0.5) * 2

        # 1. Extract features (Filtering)
        x = self.conv(depth_img_norm)
        
        # 2. Downsample to coarse grid (8 channels * 5 * 5 = 200)
        x = self.pool(x)
        
        # 3. Flatten and Compress
        cnn_feature = self.linear(self.flatten(x))  # Shape: [Batch, feature_num_cnn]

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        
        # Concatenate: Results in a balanced feature vector
        out = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = out

        return out