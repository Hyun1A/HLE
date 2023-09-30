
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
            logit = [res[i][0][hierarchy[0]] for i in range(len(res))]
            logit_hier = torch.cat(logit, dim=0)

            features = [res[i][-1] for i in range(len(res))]
            features = torch.cat(features, dim=0)

        else:
            logit_hier = res[0][0][hierarchy[0]]
            features = res[0][-1]
        
                        
        probs = F.softmax(logit_hier, dim=1)

        entropies = -(probs*torch.log(probs)).mean(1)


    return entropies.sort()


def get_gradnorm(model, x,y,hierarchy, device, batch_size):
    
    #idxlist = dataset.generate_idx(batch_size, rand=False)
    
    model.eval()
    with torch.no_grad():
        
        res = [model(x[i * batch_size:min((i + 1) * batch_size, len(x))], return_feat=True) \
            for i in range(-(-x.size(0) // batch_size))]
                
        if len(res) > 1:
            logit = [res[i][0][hierarchy[0]] for i in range(len(res))]
            logit_hier = torch.cat(logit, dim=0)

            features = [res[i][-1] for i in range(len(res))]
            features = torch.cat(features, dim=0)

        else:
            logit_hier = res[0][0][hierarchy[0]]
            features = res[0][-1]
        
                        
        probs = F.softmax(logit_hier, dim=1)
        
        gradnorm = ( (probs**2).sum(dim=1) + 1 - 2*probs[:,y[0]].squeeze(-1) ) * (features**2).sum(dim=1)


    #return gradnorm.sort()[1]
    return gradnorm.sort()