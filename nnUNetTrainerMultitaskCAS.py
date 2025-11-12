# Note: place under nnunetv2/training/nnUNetTrainer/variants/network_architecture/
# This file defines a trainer with an additional classification head.
# It keeps the original segmentation architecture by using get_network_from_plans(...)
# and wraps it to produce (seg_logits, cls_logits) during training.

from operator import itemgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch import autocast
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from typing import Any, Tuple, List, Union
import numpy as np
import pandas as pd
from time import time
from nnunet2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgenerators.utilities.file_and_folder_operations import join
from threadpoolctl import threadpool_limits


from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, write_pickle, subfiles
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
from nnunetv2.training.dataloading.nnunet_dataloader import nnUNetDataLoader

class nnUNetDatasetCAS(nnUNetDatasetNumpy):
    """ Dataset for the Multitask nnUNet with CASeg functionality.
        Returns data, seg, seg_prev, properties, CLASS where seg_prev is the segmentation from the previous stage (or None).
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
        data, seg, seg_prev, properties = super(nnUNetDatasetNumpy, self).load_case(identifier)
        class_label = self.class_labels[identifier]
        return data, seg, seg_prev, properties, class_label

    #check if preprocessing (the one that uses save_case) uses this or hte block2 dataset... Is it possile to use the block2 one for the preprocessing only?

    @staticmethod
    def save_seg(
            seg: Union[np.ndarray,List[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], #check; is output for segm a number or a ndarray too?
            output_filename_truncated: str):
        if isinstance(seg, (list, tuple)):
            #we have also class output
            segm, class_logits = seg
            np.savez_compressed(output_filename_truncated + '.npz', seg=segm, class_logits=class_logits)
        else:
            np.savez_compressed(output_filename_truncated + '.npz', seg=seg)

class nnUNetDataLoaderCAS(nnUNetDataLoader):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        class_all = np.zeros((self.batch_size,), dtype=np.int16)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties, clas = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            class_all[j] = clas
            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        class_all = torch.from_numpy(class_all).to(torch.int16)
        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {'data': data_all, 'target': seg_all, 'classes': class_all, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all, 'classes': class_all, 'keys': selected_keys}

    def determine_shapes(self):
        # load one case
        data, seg, seg_prev, properties, class_label = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        channels_seg = seg.shape[0]
        if seg_prev is not None:
            channels_seg += 1
        seg_shape = (self.batch_size, channels_seg, *self.patch_size)
        return data_shape, seg_shape



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

    def __init__(self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda")): 
        """
        cls_loss_weight: weight for classification loss 
        seg_loss_weight: weight for segmentation loss 
        n_class: number of classes for the classification head; for segm everything 
        is read from the plans as default run
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.cls_loss_weight: float = 1.0
        self.seg_loss_weight: float = 1.0
        #self.num_img_classes: int = 4  #number of classes for the classification head
        #self.aux_hidden: int = 32  #hidden dim for the aux classifier head
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

        # they need to be added in the plans.json, since method is static and has no access to self
        n_img_classes = arch_init_kwargs.get('n_img_classes', 4)
        aux_hidden = arch_init_kwargs.get('aux_hidden_dim', 32)

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

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        classes = batch['classes']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        classes = classes.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            losslist = self.loss(output, [target, classes])  # l is a list of two losses
            l = losslist[0] + losslist[1]  #combine for backprop

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy(), 'loss_seg': losslist[0].detach().cpu().numpy(), 'loss_cls': losslist[1].detach().cpu().numpy()}
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            losses_tr_seg = [None for _ in range(dist.get_world_size())]
            losses_tr_cls = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            dist.all_gather_object(losses_tr_seg, outputs['loss_seg'])
            dist.all_gather_object(losses_tr_cls, outputs['loss_cls'])
            loss_here = np.vstack(losses_tr).mean()
            loss_here_seg = np.vstack(losses_tr_seg).mean()
            loss_here_cls = np.vstack(losses_tr_cls).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            loss_here_seg = np.mean(outputs['loss_seg'])
            loss_here_cls = np.mean(outputs['loss_cls'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        # Log also segmentation and classification losses
        self.logger.log('train_losses_seg', loss_here_seg, self.current_epoch)
        self.logger.log('train_losses_cls', loss_here_cls, self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        classes = batch['classes']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        classes = classes.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            full_output = self.network(data)
            del data
            loss_list = self.loss(full_output, [target, classes])  # loss_list is a list of two losses
            l = loss_list[0] + loss_list[1]  #combine for logging

        # we only need the output with the highest output resolution (if DS enabled)
        #and after calculating the classif.loss, that output is not needed anymore
        output = full_output[0] #full_output[1] is the class, and not needed anymore
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'loss_seg': loss_list[0].detach().cpu().numpy(), 
                'loss_cls': loss_list[1].detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            losses_val_seg = [None for _ in range(world_size)]
            losses_val_cls = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            dist.all_gather_object(losses_val_seg, outputs_collated['loss_seg'])
            dist.all_gather_object(losses_val_cls, outputs_collated['loss_cls'])
            loss_here = np.vstack(losses_val).mean()
            loss_here_seg = np.vstack(losses_val_seg).mean()
            loss_here_cls = np.vstack(losses_val_cls).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])
            loss_here_seg = np.mean(outputs_collated['loss_seg'])
            loss_here_cls = np.mean(outputs_collated['loss_cls'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_losses_seg', loss_here_seg, self.current_epoch)
        self.logger.log('val_losses_cls', loss_here_cls, self.current_epoch)


    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('train_loss_seg', np.round(self.logger.my_fantastic_logging['train_losses_seg'][-1], decimals=4))
        self.print_to_log_file('train_loss_cls', np.round(self.logger.my_fantastic_logging['train_losses_cls'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss_seg', np.round(self.logger.my_fantastic_logging['val_losses_seg'][-1], decimals=4))
        self.print_to_log_file('val_loss_cls', np.round(self.logger.my_fantastic_logging['val_losses_cls'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1


#TODO:
#other versions
#training script to print losses separately byt backprop both
#inference script to output both segm and class, if it can thandle that as is
#data loader to provide both segm and class labels