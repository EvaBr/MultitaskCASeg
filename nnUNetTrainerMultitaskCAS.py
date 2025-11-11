# Note: place under nnunetv2/training/nnUNetTrainer/variants/network_architecture/
# This file defines a trainer with an additional classification head.
# It keeps the original segmentation architecture by using get_network_from_plans(...)
# and wraps it to produce (seg_logits, cls_logits) during training.

from operator import itemgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, List, Union
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
import os
from typing import List, Union, Type, Tuple
import numpy as np
import pandas as pd
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, write_pickle, subfiles
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy

class nnUNetDatasetCAS(nnUNetDatasetNumpy):
    """ Dataset for the Multitask nnUNet with CASeg functionality.
        Returns data, seg, CLASS, seg_prev, properties where seg_prev is the segmentation from the previous stage (or None).
        For loading preprocessed data + class labels.
    """
    def __init__(self, folder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        self.class_labels = None 

    def set_class_labels(self, rawfolder: str):
        csv = pd.read_csv(join(rawfolder, 'class_labels.csv'))
        csv.set_index('case_id', inplace=True)
        class_dict = dict()
        for idx, row in csv.iterrows():
            class_dict[idx] = row['class_label']
        self.class_labels = class_dict
        #self.class_labels = np.load(join(rawfolder, 'class_labels.npy'), allow_pickle=True).item()

    def load_case(self, identifier):
        img_file = super(nnUNetDatasetNumpy, self).load_case(identifier)
        if self.class_labels is None:
            return img_file #for compatibility with the  framework; first time this is called before set_class_labels (just to figure out sizes or sth?)
        class_label = self.class_labels[identifier]
        return img_file, class_label

    #check if preprocessing (the one that uses save_case) uses this or hte block2 dataset... Is it possile to use the block2 one for the preprocessing only?

    @staticmethod
    def save_seg(
            seg: Union[np.ndarray,List[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], #check; is output for segm a number or a ndarray too?
            output_filename_truncated: str
    ):
        if isinstance(seg, (list, tuple)):
            #we have also class output
            segm, class_logits = seg
            np.savez_compressed(output_filename_truncated + '.npz', seg=segm, class_logits=class_logits)
        else:
            np.savez_compressed(output_filename_truncated + '.npz', seg=seg)



class SegmentationWithImageClassifier(nn.Module):
    """
    Wraps an existing segmentation network (base_net) and returns an
    auxiliary image-level classification head output.

    Behavior:
    - forward(x) returns (seg_out, cls_logits)
    - seg_out: same as base_net(x), cls_logits: [B, n_img_classes]
    - The classifier is built by globally pooling the main segmentation logits and feeding into an FC.
      This avoids touching internal layers of base_net.
    """

    def __init__(self, base_net: nn.Module, n_seg_classes: int, n_img_classes: int = 3, aux_hidden: int = 32):
        super().__init__()
        self.base = base_net
        self.n_seg_classes = n_seg_classes
        self.n_img_classes = n_img_classes
        self.aux_hidden = aux_hidden

        # small FC to map pooled seg logits -> class logits
        # input dim for FC is n_seg_classes (we pool across spatial dims)
        self.cls_fc = nn.Sequential(
            nn.Linear(base_net.num_features_per_stage[-1], aux_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(aux_hidden, n_img_classes)
        )
        self.bottleneck_features = None
        self._hook = self.base.encoder.stages[-1].register_forward_hook(self._save_bottleneck)

    def _save_bottleneck(self, module, input, output):
        self.bottleneck_features = output 


    def forward(self, x: torch.Tensor):
        seg_out = self.base(x)
        bottleneck = self.bottleneck_features

        # Here we assume that bottleneck.dim()=5; dvs that we're working w 3d images
        pooled = F.adaptive_avg_pool3d(bottleneck, 1).view(bottleneck.size(0), -1)  # [B, C]
        cls_logits = self.cls_fc(pooled)  # [B, n_img_classes]
        
        return seg_out, cls_logits


class nnUNetTrainerMultiCASC(nnUNetTrainer):
    """
    Trainer subclass that builds the default architecture and wraps it with an auxiliary image classifier head.
    Expected data/targets:
      - data: same as nnUNet (images)
      - target: list [seg_target, class_label_tensor] where class_label_tensor has shape [B] with int labels in [0..3]
    """

    def __init__(self, *args, cls_loss_weight: float = 1.0, seg_loss_weight: float = 1.0, **kwargs):
        """
        cls_loss_weight: weight for classification loss 
        seg_loss_weight: weight for segmentation loss 
        n_class: number of classes for the classification head; for segm everything 
        is read from the plans as default run
        """
        super().__init__(*args, **kwargs)
        self.cls_loss_weight = cls_loss_weight
        self.seg_loss_weight = seg_loss_weight
        #OBS: here we dont enable deep supervision for classification, since in nnunet deep 
        # supervision is only done in the decoder part...
        #self.deep_supervision_class = False 

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Builds the base segmentation net (via get_network_from_plans) and wraps it into
        SegmentationWithImageClassifier, adding a classifier head.
        num_output_channels is the segmentation output channels (e.g., 3).
        """
        # build the original segmentation network (this uses the existing plans resolution)
        base_net = get_network_from_plans(architecture_class_name,
                                          arch_init_kwargs,
                                          arch_init_kwargs_req_import,
                                          num_input_channels,
                                          num_output_channels,
                                          allow_init=True,
                                          deep_supervision=enable_deep_supervision)

        n_img_classes = arch_init_kwargs.get('n_img_classes', 4)
        aux_hidden = arch_init_kwargs.get('aux_hidden_dim', 32)

        # copy arch kwargs and remove our extra keys so base constructor isn't confused
        #arch_kwargs_for_base = dict(arch_init_kwargs)
        #for k in ('n_img_classes', 'aux_hidden_dim'):
        #    arch_kwargs_for_base.pop(k, None)

        wrapped = SegmentationWithImageClassifier(base_net, n_seg_classes=num_output_channels,
                                                  n_img_classes=n_img_classes,
                                                  aux_hidden=aux_hidden)
        return wrapped

    def _build_loss(self):
        """
        Build segmentation loss using parent's implementation and then return a combined loss
        that also adds a CrossEntropy classification loss.
        """
        # build segmentation loss exactly as parent does (it returns either a loss or DeepSupervisionWrapper etc.)
        loss_seg = super()._build_loss()
        loss_cls = nn.CrossEntropyLoss()
        #combined = CombinedSegAndClsLoss(seg_loss, cls_loss_weight=self.cls_loss_weight, seg_loss_weight=self.seg_loss_weight)
        
        if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()
                weights_seg = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
                weights_seg[-1] = 1e-16  # don't use lowest resolution output
                weights_seg = weights_seg / weights_seg.sum()

                #if self.deep_supervision_class:
                #    weights_cls = weights_seg.copy()
                #else:
                #    weights_cls = np.array([1] + [0] * (len(deep_supervision_scales) - 1)) #use only highest res
                #loss_cls = DeepSupervisionWrapper(loss_cls, weights_cls)

                # now wrap the loss
                loss_seg = DeepSupervisionWrapper(loss_seg, weights_seg)

        #return list of two, so you can plot and monitor them individually
        return lambda x,y: [self.seg_loss_weight * loss_seg(x[0], y[0]), self.cls_loss_weight * loss_cls(x[1], y[1])] 
    
    def initialize(self):
        # call parent initialize which will call our build_network_architecture above
        super().initialize()

    def get_training_transforms(self):
        # use parent's transforms
        return ComposeTransforms(itemgetter(0), super().get_training_transforms()) #the first thing is image, so only do transforms on that
    def get_validation_transforms(self):
        return ComposeTransforms(itemgetter(0), super().get_validation_transforms())

#TODO: 
#other versions
#training script to print losses separately byt backprop both
#inference script to output both segm and class, if it can thandle that as is
#data loader to provide both segm and class labels