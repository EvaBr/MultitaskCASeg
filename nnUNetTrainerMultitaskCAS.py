# Note: place under nnunetv2/training/nnUNetTrainer/variants/network_architecture/
# This file defines a trainer with an additional classification head.
# It keeps the original segmentation architecture by using get_network_from_plans(...)
# and wraps it to produce (seg_logits, cls_logits) during training.
import multiprocessing
from pathlib import Path
import warnings
import itertools
from operator import itemgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch import autocast
from typing import Any, Tuple, List, Union
import numpy as np
import pandas as pd
from time import time, sleep
from queue import Queue
from threading import Thread
from tqdm import tqdm

from acvl_utils.cropping_and_padding.padding import pad_nd_image
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from threadpoolctl import threadpool_limits

from nnunet2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy, infer_dataset_class
from nnunetv2.training.dataloading.nnunet_dataloader import nnUNetDataLoader

from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, save_pickle
from nnunetv2.configuration import default_num_processes

from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import resample_and_save, convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

#nnunetpredictor:
#predict_logits_from_preprocessed_data: prediction += self.predict_sliding_window_return_logits(data)

#these three are urgent for the training to work! The rest are important for inference standalone
#_internal_maybe_mirror_and_predict: prediction+=self.network(data)...
#_internal_predict_sliding_window_return_logits: prediction = _internal_maybe_mirror_and_predict(...)
#predict_sliding_window_return_logits: predicted_logits=_internal_predict_sliding_window_return_logits

#predict_from_files_sequential: prediction = predict_logits_from_preprocessed_data(...)
#predict_from_data_iterator: prediction = predict_logits_from_preprocessed_data(...)

class MynnUNetPredictor(nnUNetPredictor):
    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction, class_prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes))[0], axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction, class_prediction
    
    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
            predicted_class_logits = torch.zeros((data.shape[0], self.configuration_manager.num_img_classes), 
                                                 dtype=torch.half, device=results_device)
            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction, class_prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                    print("class placeholder shape:", predicted_class_logits.shape,"class pred shape:", class_prediction.shape)
                    print("predicted_class_logits[sl[0]].shape=", predicted_class_logits[sl[0]].shape)
                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    predicted_class_logits[sl[0]] += class_prediction
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits, predicted_class_logits
    

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits, predicted_classes = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits, predicted_classes = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits, predicted_classes = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits, predicted_classes

def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], 
                                  predicted_class_logits: Union[np.ndarray, torch.Tensor],
                                  properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False,
                                  num_threads_torch: int = default_num_processes):

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities, num_threads_torch=num_threads_torch
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret
    np.savez_compressed(output_file_truncated + '_class.npz', clas=predicted_class_logits)
    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)



class nnUNetDatasetCAS(nnUNetDatasetNumpy):
    """ Dataset for the Multitask nnUNet with CASeg functionality.
        Returns data, seg, seg_prev, properties, CLASS where seg_prev is the segmentation from the previous stage (or None).
        For loading preprocessed data + class labels.
    """
    def __init__(self, folder: str, rawfolder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        
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
        self.dataset_class = nnUNetDatasetCAS

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = nnUNetDatasetCAS

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderCAS(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoaderCAS(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


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
    
    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = MynnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties, _ = dataset_val.load_case(k)

                # we do [:] to convert blosc2 to numpy
                data = data[:]

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction, class_prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()
                class_prediction = class_prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async( #TODO export_prediciton_from _logits should save even class!
                        export_prediction_from_logits, (
                            (prediction, class_prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        # next stage may have a different dataset class, do not use self.dataset_class
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = dataset_class(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file_truncated = join(output_folder, k)

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file_truncated, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json,
                                 default_num_processes,
                                 dataset_class),
                            )
                        ))
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            #TODO addc ompute metrics for classification
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            classmetrics = compute_classification_metrics_on_folder(
                join(self.preprocessed_dataset_folder_base, 'class_labels.csv'),
                validation_output_folder,
                self.dataset_json["file_ending"])
            
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)
            self.print_to_log_file("Mean classification accuracy: ", (classmetrics['classification_accuracy']),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()


def compute_classification_metrics_on_folder(class_labels_csv, validation_output_folder, file_ending):
    csv = pd.read_csv(class_labels_csv, index_col=0)
    allfiles = [fl for fl in Path(validation_output_folder).glob(f'*.npz') if fl.name[:-4] in csv.index]
    all_class_predictions = [np.argmax(np.load(fl)['clas']) for fl in allfiles]
    #now compare output with GT.
    gt_classes = [csv.loc[fl.name[:-4], 'class_label'] for fl in allfiles]
    all_class_predictions = np.array(all_class_predictions)
    gt_classes = np.array(gt_classes)

    # Compute metrics
    accuracy = np.mean(all_class_predictions == gt_classes)
    return {"classification_accuracy": accuracy}


#TODO:
#other versions
#DONE training script to print losses separately but backprop both
#inference script to output both segm and class, if it can thandle that as is
#DONE data loader to provide both segm and class labels