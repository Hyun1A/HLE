# Official Implementation of PL-FMS (Online Continual Learning on Hierarchical Label Expansion)

**Online Continual Learning on Hierarchical Label Expansion**
<br>Byung Hyun Lee<sup>\*</sup>, Okchul Jung<sup>\*</sup>, Jonghyun Choi<sup>&dagger;</sup>, Se Young Chun<sup>&dagger;</sup><br>
(\* indicates equal contribution, &dagger; indicates corresponding author)

ICCV 2023 [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Online_Continual_Learning_on_Hierarchical_Label_Expansion_ICCV_2023_paper.pdf)]<br>


## Overview
### Comparison to previous setups
<img src="./figs/figure1_setup_comparison.png" width="700">
(a) conventional task-free online CL, (b) blurry task-free online CL setup, and (c) proposed Hierarchical Label Expansion (HLE) CL setup


### Scenarios with hierarchial label expansion
<img src="./figs/figure2_expansion_scenario.png" width="700">
(a) single-depth (single-label & dual-label) and (b) multi-depth scenarios

### PL-FMS
<img src="./figs/figure3_method.png" width="700">

### Abstract
Continual learning (CL) enables models to adapt to new tasks and environments without forgetting previously learned knowledge. While current CL setups have ignored the relationship between labels in the past task and the new task with or without small task overlaps, real-world scenarios often involve hierarchical relationships between old and new tasks, posing another challenge for traditional CL approaches. To address this challenge, we propose a novel multi-level hierarchical class incremental task configuration with an online learning constraint, called hierarchical label expansion (HLE). Our configuration allows a network to first learn coarse-grained classes, with data labels continually expanding to more fine-grained classes in various hierarchy depths. To tackle this new setup, we propose a rehearsal-based method that utilizes hierarchy-aware pseudo-labeling to incorporate hierarchical class information. Additionally, we propose a simple yet effective memory management and sampling strategy that selectively adopts samples of newly encountered classes. Our experiments demonstrate that our proposed method can effectively use hierarchy on our HLE setup to improve classification accuracy across all levels of hierarchies, regardless of depth and class imbalance ratio, outperforming prior arts by significant margins while also outperforming them on the conventional disjoint, blurry and i-Blurry CL setups.

### Results
<img src="./figs/figure4_depth1.png" width="700">


## Getting Started
### Experiment environment
**OS**: Ubuntu 20.04 LTS

**GPU**: Geforce RTX 3090 with CUDA 11.1

**Python**: 3.8.15

To set up the python environment for running the code, we provide requirements.txt that can be installed using the command
<pre>
pip install -r requirements.txt
</pre>

## Running Experiments

### Downloading the Datasets
CIFAR100 can be downloaded by running the corresponding scripts in the `./data/datasets/` directory.
ImageNet dataset can be downloaded from [ImageNet](https://www.kaggle.com/c/imagenet-object-localization-challenge). Stanford cars can be downloaded from [StanfordCars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset). iNaturalist-19 can be downloaded from [iNaturalist19](https://github.com/visipedia/inat_comp/tree/master/2019#Data)

### Experiments Using Shell Script
Experiments for the implemented methods can be run by:

**Single-label secnario**
<pre>
cd single_depth
sh scripts/single_label.sh {Dataset} {Method}
</pre>

**Dual-label secnario**
<pre>
cd single_depth
sh scripts/dual_label.sh {Dataset} {Method}
</pre>

**Multi-depth secnario**
<pre>
cd multi_depth
sh scripts/multi_depth.sh {Dataset} {Method}
</pre>


You can change the arguments for different experiments.
- `Dataset`: Dataset to use in experiment. Supported datasets are [cifar100, stanford_car, imagenet] for single-label & dual-label scenarios and [cifar100, inat19] for multi-depth scenario. 
- `MODE`: CL method to be applied. Methods implemented in this version are: [pl_fms, clib, er, ewc++, bic, mir, gdumb, rm]

## Citation
If you used our code for HLE continual learning setup, please cite our paper.
<pre>
@InProceedings{lee2023hle,
  author    = {Lee, Byung Hyun and Jung, Okchul and Choi, Jonghyun and Chun, Se Young},
  title     = {Online Continual Learning on Hierarchical Label Expansion},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
  pages     = {11761-11770}
} 
</pre>

## License
This code is implemented based on [the code of i-Blurry CL](https://github.com/naver-ai/i-Blurry). More details for our code can be found from this base code.

```
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
