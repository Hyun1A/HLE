"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from collections import Counter
import torch



class FixMatchModule(object):
    r"""
    Dynamic thresholding module from `FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling
    <https://arxiv.org/abs/2110.08263>`_. At time :math:`t`, for each category :math:`c`,
    the learning status :math:`\sigma_t(c)` is estimated by the number of samples whose predictions fall into this class
    and above a threshold (e.g. 0.95). Then, FlexMatch normalizes :math:`\sigma_t(c)` to make its range between 0 and 1
    .. math::
        \beta_t(c) = \frac{\sigma_t(c)}{\underset{c'}{\text{max}}~\sigma_t(c')}.
    The dynamic threshold is formulated as
    .. math::
        \mathcal{T}_t(c) = \mathcal{M}(\beta_t(c)) \cdot \tau,
    where \tau denotes the pre-defined threshold (e.g. 0.95), :math:`\mathcal{M}` denotes a (possibly non-linear)
    mapping function.
    Args:
        threshold (float): The pre-defined confidence threshold
        warmup (bool): Whether perform threshold warm-up. If True, the number of unlabeled data that have not been
            used will be considered when normalizing :math:`\sigma_t(c)`
        mapping_func (callable): An increasing mapping function. For example, this function can be (1) concave
            :math:`\mathcal{M}(x)=\text{ln}(x+1)/\text{ln}2`, (2) linear :math:`\mathcal{M}(x)=x`,
            and (3) convex :math:`\mathcal{M}(x)=2/2-x`
        num_classes (int): Number of classes
        n_unlabeled_samples (int): Size of the unlabeled dataset
        device (torch.device): Device
    """

    def __init__(self, threshold, warmup, mapping_func, device, depth=0):
        self.threshold = threshold
        self.warmup = warmup
        self.mapping_func = mapping_func
        self.device = device
        self.depth = depth
        
        self.predicted_hist = [torch.tensor([]) for h in range(depth+1)]
    

    def get_threshold(self, pseudo_labels, hierarchy):
        """Calculate and return dynamic threshold"""
        return self.threshold
        

    def update(self, pseudo_labels, hierarchy, weight=1.0):
        """Update the learning status
        Args:
            idxes (tensor): Indexes of corresponding samples
            selected_mask (tensor): A binary mask, a value of 1 indicates the prediction for this sample will be updated
            pseudo_labels (tensor): Network predictions
        """
        pass
    
    def add_new_class(self, hierarchy):
        self.predicted_hist[hierarchy] = torch.cat([self.predicted_hist[hierarchy], torch.tensor([0.])], dim=0)
        
        
class FlexMatchModule(object):
    r"""
    Dynamic thresholding module from `FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling
    <https://arxiv.org/abs/2110.08263>`_. At time :math:`t`, for each category :math:`c`,
    the learning status :math:`\sigma_t(c)` is estimated by the number of samples whose predictions fall into this class
    and above a threshold (e.g. 0.95). Then, FlexMatch normalizes :math:`\sigma_t(c)` to make its range between 0 and 1
    .. math::
        \beta_t(c) = \frac{\sigma_t(c)}{\underset{c'}{\text{max}}~\sigma_t(c')}.
    The dynamic threshold is formulated as
    .. math::
        \mathcal{T}_t(c) = \mathcal{M}(\beta_t(c)) \cdot \tau,
    where \tau denotes the pre-defined threshold (e.g. 0.95), :math:`\mathcal{M}` denotes a (possibly non-linear)
    mapping function.
    Args:
        threshold (float): The pre-defined confidence threshold
        warmup (bool): Whether perform threshold warm-up. If True, the number of unlabeled data that have not been
            used will be considered when normalizing :math:`\sigma_t(c)`
        mapping_func (callable): An increasing mapping function. For example, this function can be (1) concave
            :math:`\mathcal{M}(x)=\text{ln}(x+1)/\text{ln}2`, (2) linear :math:`\mathcal{M}(x)=x`,
            and (3) convex :math:`\mathcal{M}(x)=2/2-x`
        num_classes (int): Number of classes
        n_unlabeled_samples (int): Size of the unlabeled dataset
        device (torch.device): Device
    """

    def __init__(self, threshold, warmup, mapping_func, device, depth=0):
        self.threshold = threshold
        self.warmup = warmup
        self.mapping_func = mapping_func
        self.device = device
        self.depth = depth
        
        self.predicted_hist = [torch.tensor([]).to(self.device) for h in range(depth+1)]
    

    def get_threshold(self, cls_prob):
        """Calculate and return dynamic threshold"""
        return self.threshold * cls_prob
        
        #print(hierarchy)
        #print(type(hierarchy))
        
        #self.predicted_hist[hierarchy].max()
        
        status = self.predicted_hist[hierarchy] / (self.predicted_hist[hierarchy].max() + 1e-10)
        dynamic_threshold = self.threshold * self.mapping_func(status[pseudo_labels])

        #print('dynamic_threshold:', dynamic_threshold)
        
        return dynamic_threshold
        

    def update(self, pseudo_labels, weight):
        """Update the learning status
        Args:
            idxes (tensor): Indexes of corresponding samples
            selected_mask (tensor): A binary mask, a value of 1 indicates the prediction for this sample will be updated
            pseudo_labels (tensor): Network predictions
        """
        
        for h in range(self.depth+1):
            labels,counts = torch.unique(pseudo_labels[h], return_counts=True)
            
            if len(pseudo_labels[h]) > 0:
                #print('predicted_hist:', self.predicted_hist[h][labels])
                #print('weight:', weight[h])
                #print('labels:', labels)
                #print('counts:', counts)
                #print()
                
                self.predicted_hist[h][labels] += weight[h][0]*counts

    
    def add_new_class(self, hierarchy):
        self.predicted_hist[hierarchy] = torch.cat([self.predicted_hist[hierarchy], torch.tensor([0.]).to(self.device)], dim=0).to(self.device)
