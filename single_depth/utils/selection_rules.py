
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns points that minimizes the maximum distance of any point to a center.
Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017
Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).
Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

def get_entropy(model, x,y,hierarchy, device, batch_size):
       
    model.eval()
    with torch.no_grad():
        res = [model(x[i * batch_size:min((i + 1) * batch_size, len(x))], return_feat=True) \
            for i in range(-(-x.size(0) // batch_size))]
        
        if len(res) > 1:

            logit_sup = [res[i][0] for i in range(len(res))]
            logit_sub = [res[i][1] for i in range(len(res))]
            features = [res[i][2] for i in range(len(res))]

            logit_sup = torch.cat(logit_sup, dim=0)
            logit_sub = torch.cat(logit_sub, dim=0)
            features = torch.cat(features, dim=0)

        else:
            logit_sup, logit_sub, features = res[0]
        
                        
        if hierarchy[0].item() == 0:
            probs = F.softmax(logit_sup, dim=1)
        else:
            probs = F.softmax(logit_sub, dim=1)       
    
                
    entropies = -(probs*torch.log(probs)).sum(1)
    
    """
    print('-------- entropies --------')
    print(y)
    print(hierarchy)
    print(entropies.sort()[0])
    print(entropies.sort()[1])
    print('-------- --------- --------')
    """

    return entropies.sort()




def get_gradnorm(model, x,y,hierarchy, device, batch_size):
    
    #idxlist = dataset.generate_idx(batch_size, rand=False)
    
    model.eval()
    with torch.no_grad():
        
        res = [model(x[i * batch_size:min((i + 1) * batch_size, len(x))], return_feat=True) \
            for i in range(-(-x.size(0) // batch_size))]
        
        if len(res) > 1:

            logit_sup = [res[i][0] for i in range(len(res))]
            logit_sub = [res[i][1] for i in range(len(res))]
            features = [res[i][2] for i in range(len(res))]

            logit_sup = torch.cat(logit_sup, dim=0)
            logit_sub = torch.cat(logit_sub, dim=0)
            features = torch.cat(features, dim=0)

        else:
            logit_sup, logit_sub, features = res[0]
        
                        
        if hierarchy[0].item() == 0:
            probs = F.softmax(logit_sup, dim=1)
        else:
            probs = F.softmax(logit_sub, dim=1)                        
        
    gradnorm = ( (probs**2).sum(dim=1) + 1 - 2*probs[:,y[0]].squeeze(-1) ) * (features**2).sum(dim=1)


    return gradnorm.sort()

def get_random(model, x,y,hierarchy, device, batch_size):
    

    return 0, torch.randperm(y.size(0)).to(device)