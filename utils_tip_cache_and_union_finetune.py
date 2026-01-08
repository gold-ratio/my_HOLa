"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from code import interact
from fileinput import filename
from locale import normalize
import os
import torch
import pickle
import numpy as np
import scipy.io as sio
import json

from torchvision.transforms import Resize, CenterCrop

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from hicodet.hicodet import HICODet
from hico_text_label import hico_unseen_index, HOI_TO_AO, ACT_IDX_TO_ACT_NAME, obj_to_name, OBJ_IDX_TO_COCO_ID, MAP_AO_TO_HOI
import sys
sys.path.append('../pocket/pocket')
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

sys.path.append('detr')
import detr.datasets.transforms_clip as T
import pdb
import copy 
import pickle
import torch.nn.functional as F
import clip
from util import box_ops
from PIL import Image
from hicodet.static_hico import HICO_INTERACTIONS
from hico_text_label import HICO_INTERACTIONS, hico_unseen_index 
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#将一个批次的数据项从列表中转换成两个独立的列表
def custom_collate(batch):
    images = []
    targets = []
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, clip_model_name, zero_shot=False, zs_type='rare_first', num_classes=600, detr_backbone="R50", syn = None): ##ViT-B/16, ViT-L/14@336px
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        assert clip_model_name in ['ViT-L/14@336px', 'ViT-B/16',  'ViT-B/32']
        self.clip_model_name = clip_model_name
        if self.clip_model_name == 'ViT-B/16' or self.clip_model_name == 'ViT-B/32':
            self.clip_input_resolution = 224
        elif self.clip_model_name == 'ViT-L/14@336px':
            self.clip_input_resolution = 336
        
        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            if syn is not None:
                self.syndataset = HICODet(
                root=os.path.join("./syn_data", syn[0]),
                anno_file=os.path.join("./syn_data", syn[1]),
                target_transform=pocket.ops.ToTensor(input_format='dict')
                )
                self.dataset = self.dataset + self.syndataset

        # add clip normalization
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
            ])
            self.clip_transforms = T.Compose([
                T.IResize([self.clip_input_resolution,self.clip_input_resolution]),
            ])
        else:   
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
            ])
            self.clip_transforms = T.Compose([
                T.IResize([self.clip_input_resolution,self.clip_input_resolution]),
            ])
        
        self.partition = partition
        self.name = name
        self.count=0
        self.zero_shot = zero_shot
        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_type = zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]

        device = "cuda"
        # _, self.process = clip.load('ViT-B/16', device=device)
        print(self.clip_model_name)
        _, self.process = clip.load(self.clip_model_name, device=device)

        self.keep = [i for i in range(len(self.dataset))]

        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_keep = []
            self.remain_hoi_idx = [i for i in np.arange(600) if i not in self.filtered_hoi_idx]

            for i in self.keep:
                (image, target), filename = self.dataset[i]
                # if 1 in target['hoi']:
                #     pdb.set_trace()
                mutual_hoi = set(self.remain_hoi_idx) & set([_h.item() for _h in target['hoi']])
                if len(mutual_hoi) != 0:
                    self.zs_keep.append(i)
            self.keep = self.zs_keep
            if syn is None:
                num_object_cls = self.dataset.num_object_cls
                class_corr = self.dataset.class_corr
            else:
                num_object_cls = self.syndataset.num_object_cls
                class_corr = self.syndataset.class_corr   

            self.zs_object_to_target = [[] for _ in range(num_object_cls)]
            if num_classes == 600:
                for corr in class_corr:
                    if corr[0] not in self.filtered_hoi_idx:
                        self.zs_object_to_target[corr[1]].append(corr[0])
            else:
                for corr in class_corr:
                    if corr[0] not in self.filtered_hoi_idx:
                        self.zs_object_to_target[corr[1]].append(corr[2])        

    def __len__(self):
        return len(self.keep)

    # train detr with roi
    def __getitem__(self, i):
        (image, target), filename = self.dataset[self.keep[i]]
        # (image, target), filename = self.dataset[i]
        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            _boxes_h, _boxes_o, _hoi, _object, _verb = [], [], [], [], []
            for j, hoi in enumerate(target['hoi']):
                if hoi in self.filtered_hoi_idx:
                    # pdb.set_trace()
                    continue
                _boxes_h.append(target['boxes_h'][j])
                _boxes_o.append(target['boxes_o'][j])
                _hoi.append(target['hoi'][j])
                _object.append(target['object'][j])
                _verb.append(target['verb'][j])           
            target['boxes_h'] = torch.stack(_boxes_h)
            target['boxes_o'] = torch.stack(_boxes_o)
            target['hoi'] = torch.stack(_hoi)
            target['object'] = torch.stack(_object)
            target['verb'] = torch.stack(_verb)
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w])

        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')
            ## TODO add target['hoi']
        
        image, target = self.transforms(image, target)
        image_clip, target = self.clip_transforms(image, target)  
        image, _ = self.normalize(image, None)
        image_clip, target = self.normalize(image_clip, target)
        target['filename'] = filename

        return (image,image_clip), target

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def get_region_proposals(self, results,image_h, image_w):
        human_idx = 0
        min_instances = 3
        max_instances = 15
        region_props = []
        bx = results['ex_bbox']
        sc = results['ex_scores']
        lb = results['ex_labels']
        hs = results['ex_hidden_states']
        is_human = lb == human_idx
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human.sum(); n_object = len(lb) - n_human
        # Keep the number of human and object instances in a specified interval
        device = torch.device('cpu')
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            keep_h = hum

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            keep_o = obj

        keep = torch.cat([keep_h, keep_o])

        boxes=bx[keep]
        scores=sc[keep]
        labels=lb[keep]
        hidden_states=hs[keep]
        is_human = labels == human_idx
            
        n_h = torch.sum(is_human); n = len(boxes)
        # Permute human instances to the top
        if not torch.all(labels[:n_h]==human_idx):
            h_idx = torch.nonzero(is_human).squeeze(1)
            o_idx = torch.nonzero(is_human == 0).squeeze(1)
            perm = torch.cat([h_idx, o_idx])
            boxes = boxes[perm]; scores = scores[perm]
            labels = labels[perm]; unary_tokens = unary_tokens[perm]
        # Skip image when there are no valid human-object pairs
        if n_h == 0 or n <= 1:
            print(n_h, n)
        # Get the pairwise indices
        x, y = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )
        # pdb.set_trace()
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
        sub_boxes = boxes[x_keep]
        obj_boxes = boxes[y_keep]
        lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
        rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
        union_boxes = torch.cat([lt,rb],dim=-1)
        sub_boxes[:,0].clamp_(0, image_w)
        sub_boxes[:,1].clamp_(0, image_h)
        sub_boxes[:,2].clamp_(0, image_w)
        sub_boxes[:,3].clamp_(0, image_h)

        obj_boxes[:,0].clamp_(0, image_w)
        obj_boxes[:,1].clamp_(0, image_h)
        obj_boxes[:,2].clamp_(0, image_w)
        obj_boxes[:,3].clamp_(0, image_h)

        union_boxes[:,0].clamp_(0, image_w)
        union_boxes[:,1].clamp_(0, image_h)
        union_boxes[:,2].clamp_(0, image_w)
        union_boxes[:,3].clamp_(0, image_h)
        return sub_boxes, obj_boxes, union_boxes

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

from torch.cuda import amp
class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, start_epoch = 0, writer=None, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes
        self._start_epoch = start_epoch
        # ⭐ TensorBoard
        self.writer = writer
        self.global_step = start_epoch * len(dataloader)

    def _on_each_iteration(self):
        # with amp.autocast(enabled=True):
        loss_dict = self._state.net(*self._state.inputs, targets=self._state.targets)
        # Check NaN on scalar losses only (avoid hist tensors)
        scalar_vals = [v for v in loss_dict.values() if torch.is_tensor(v) and v.dim()==0]
        if len(scalar_vals)>0 and torch.isnan(torch.stack([v for v in scalar_vals])).any():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")
        # Sum only true loss terms: tensors that require gradients
        loss_terms = [v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad]
        if len(loss_terms) > 0:
            self._state.loss = sum(loss_terms)
        else:
            param_device = next(self._state.net.parameters()).device
            self._state.loss = torch.tensor(0., device=param_device)
        # print("final loss ", self._state.loss, epoch, self._start_epoch)
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()
        # ===============================
        # ⭐ TensorBoard logging（核心）
        # ===============================
        if self.writer is not None and self._rank == 0:
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    if v.dim() == 0:
                        self.writer.add_scalar(
                            f'Loss/{k}',
                            v.item(),
                            self.global_step
                        )
                    elif v.dim() == 1:
                        # Log distributions (e.g., alpha_values/beta_values)
                        self.writer.add_histogram(
                            f'Dist/{k}', v.detach().cpu(), self.global_step
                        )

            self.writer.add_scalar(
                'Loss/total',
                self._state.loss.item(),
                self.global_step
            )

            lr = self._state.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('LR', lr, self.global_step)
        # ===============================
        # 控制台打印训练进度
        # ===============================
        if self._rank == 0 and self.global_step % 50 == 0:  # 每50步打印一次
            display_items = [f"{k}: {v.item():.4f}" for k, v in loss_dict.items() if torch.is_tensor(v) and v.dim()==0]
            print(f"[Step {self.global_step}] Loss total: {self._state.loss.item():.4f}, "
                  + ", ".join(display_items)
                  + f", LR: {lr:.6f}")

        self.global_step += 1

    @torch.no_grad()
    def test_hico(self, dataloader, args=None):
        net = self._state.net
        net.eval()
        dataset = dataloader.dataset.dataset
        interaction_to_verb = torch.as_tensor(dataset.interaction_to_verb)

        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        tgt_num_classes = 600
        
        num_gt = dataset.anno_interaction if args.dataset == "hicodet" else None
        meter = DetectionAPMeter(
            tgt_num_classes, nproc=1,
            num_gt=num_gt,
            algorithm='11P'
        )
        count = 0
        seen_conf = {}
        unseen_conf = {}

        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            outputs = net(inputs,batch[1])
            # Skip images without detections
            if outputs is None or len(outputs) == 0:
                continue

            for output, target in zip(outputs, batch[-1]):
                count += 1
                output = pocket.ops.relocate_to_cpu(output, ignore=True)
                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
                objects = output['objects']
                scores = output['scores']
                verbs = output['labels']
                if net.module.num_classes==117 or net.module.num_classes==407:
                    interactions = conversion[objects, verbs]
                else:
                    interactions = verbs

                # Recover target box scale
                gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_hoi = interactions.unique()
                
                for hoi_idx in unique_hoi:
                    gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )
                        # all_det_idxs.append(det_idx)
                gt_ind = torch.nonzero(labels).squeeze(1)
                verb_gt = verbs[gt_ind]
                obj_gt = objects[gt_ind]
                score_pred = scores[gt_ind]
                hoi_gt = [MAP_AO_TO_HOI[verb_gt[i], obj_gt[i]] for i in range(len(verb_gt))]
                for iidxx, temp_hoii in enumerate(hoi_gt):
                    if temp_hoii in hico_unseen_index[args.zs_type]:
                        if temp_hoii not in unseen_conf:
                            unseen_conf[temp_hoii] = []
                        unseen_conf[temp_hoii].append(score_pred[iidxx].item())
                    else:
                        if temp_hoii not in seen_conf:
                            seen_conf[temp_hoii] = []
                        seen_conf[temp_hoii].append(score_pred[iidxx].item())
            
                if args.vis_img == True:
                    save_path = os.path.join(args.vis_img_path)
                    if os.path.exists(save_path):
                        pass
                    else:
                        os.makedirs(save_path)
                    
                    save_path = os.path.join(save_path, batch[1][0]['filename'])

                    img_result = cv2.imread(os.path.join("../data/hico_20160224_det/images/test2015", batch[1][0]['filename'])).astype(np.float32)
                    
                    h, w, c = img_result.shape
                    h_ratio = h/target['size'][0]
                    w_ratio = w/target['size'][1]
                    
                    int_score, int_ind = scores.topk(7)
                    idx = 0
                    for temp_score, temp_ind in zip(int_score, int_ind):
                        idx += 1
                        if idx > 6:
                            continue

                        color = self.random_color()

                        # human
                        x1, y1, x2, y2 = boxes_h[temp_ind]
                        x1 *= w_ratio
                        x2 *= w_ratio
                        y1 *= h_ratio
                        y2 *= h_ratio
                        h_name = "person"+str(idx)
                        cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(img_result, '%s' % (h_name), (int(x1), int(y2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
                        
                        # object& action
                        x1, y1, x2, y2 = boxes_o[temp_ind]
                        x1 *= w_ratio
                        x2 *= w_ratio
                        y1 *= h_ratio
                        y2 *= h_ratio
                        ao_name = HICO_INTERACTIONS[int(interactions[temp_ind])]['object']+str(idx)
                        cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(img_result, '%s' % (ao_name), (int(x1), int(y2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
                        # action
                        i_name = HICO_INTERACTIONS[int(interactions[temp_ind])]['action'] + str(idx)
                        if temp_ind in gt_ind:
                            i_name = i_name + " GT"
                        if int(interactions[temp_ind]) in hico_unseen_index[args.zs_type]:
                            i_name = i_name + " US"
                        cv2.putText(img_result, '%s' % (i_name +": " +str(round(temp_score.item(), 3))),
                                (10, int(50 * idx + 50)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)                    

                    cv2.imwrite( save_path, img_result)
                meter.append(scores, interactions, labels)   # scores human*object*verb, interaction（600), labels
            
        return meter.eval()

    def random_color(self):
        rdn = random.randint(1, 1000)
        b = int(rdn * 997) % 255
        g = int(rdn * 4447) % 255
        r = int(rdn * 6563) % 255
        return b, g, r
    
    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache', args = None):
        net = self._state.net
        net.eval()
        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs, batch[1])

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            if args.vis_img == True:
                save_path = os.path.join(args.vis_img_path)
                if os.path.exists(save_path):
                    pass
                else:
                    os.makedirs(save_path)
                
                save_path = os.path.join(save_path, batch[1][0]['filename'])

                img_result = cv2.imread(os.path.join("../data/mscoco2014/val2014", batch[1][0]['filename'])).astype(np.float32)

            idx = -1
            cnt = -1
            for bh, bo, scr, acti in zip(boxes_h, boxes_o, scores, actions):
                idx += 1
                a_name = dataset.actions[acti].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = scr.item()
                result['_'.join(a_name)] = bo.tolist() + [scr.item()]
                all_results.append(result)

                if args.vis_img == True and scr >= scores.topk(4).values[-1] :  
                    cnt += 1
                    color = self.random_color()

                    x1, y1, x2, y2 = bh
                    h_name = "person"
                    cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(img_result, '%s' % (h_name), (int(x1), int(y2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
                    
                    # object& action
                    x1, y1, x2, y2 = bo
                    o_name = obj_to_name[output['objects'][idx]]
                    cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(img_result, '%s' % (o_name), (int(x1), int(y2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
                    # action

                    cv2.putText(img_result, '%s' % (a_name[0] +": " +str(round(scr.item(), 3))),
                            (10, int(50 * cnt + 50)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)                    

                    cv2.imwrite( save_path, img_result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir) 
        print('saving cache.pkl to', cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
