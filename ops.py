"""
Opearations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops

from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment

import sys
sys.path.append('detr')
from util.box_ops import generalized_box_iou
from scipy.ndimage import gaussian_filter

class BalancedBoxSampler:
    def __init__(self, threshold: float = .2, perc: float = .8) -> None:
        self.threshold = threshold
        self.perc = perc

    def __call__(self, scores: Tensor, number: int) -> Tensor:
        """
        Parameters:
        -----------
        scores: Tensor
            (N,) The confidence scores for a set of bounding boxes
        number: int
            The number of boxes to sample

        Returns:
        --------
        sampled_high: Tensor
            Indices of sampled high-confidence examples
        sampled_low: Tensor
            Indices of sampled low-confidence examples
        """
        idx_high = torch.nonzero(scores >= self.threshold).squeeze(1)
        idx_low = torch.nonzero(scores < self.threshold).squeeze(1)

        n_high = int(number * self.perc)
        # Protect against not enough high-confidence examples
        n_high = min(idx_high.numel(), n_high)
        n_low = number - n_high
        # Protect against not enough low-confidence examples
        n_low = min(idx_low.numel(), n_low)

        perm_high = torch.randperm(idx_high.numel(), device=idx_high.device)[:n_high]
        perm_low = torch.randperm(idx_low.numel(), device=idx_low.device)[:n_low]

        sampled_high = idx_high[perm_high]
        sampled_low = idx_low[perm_low]

        return sampled_high, sampled_low

class BoxPairCoder:
    def __init__(self,
        weights: Optional[List[float]] = None,
        bbox_xform_clip: float = math.log(1000. / 16)
    ) -> None:
        if weights is None:
            weights = [10., 10., 5., 5.]
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, props_h: Tensor, props_o: Tensor, target_h: Tensor, target_o: Tensor) -> Tensor:
        """
        Compute the regression targets based on proposed boxes pair and target box pairs.
        NOTE that all boxes are presumed to have been normalised by image width and height
        and are in (c_x, c_y, w, h) format.

        Parameters:
        -----------
        props_h: Tensor
            (N, 4) Human box proposals
        props_o: Tensor
            (N, 4) Object box proposals
        target_h: Tensor
            (N, 4) Human box targets
        target_o: Tensor
            (N, 4) Object box targets

        Returns:
        --------
        box_deltas: Tensor
            (N, 8) Regression targets for proposed box pairs
        """
        wx, wy, ww, wh = self.weights
        dx_h = wx * (target_h[:, 0] - props_h[:, 0])
        dy_h = wy * (target_h[:, 1] - props_h[:, 1])
        dw_h = ww * torch.log(target_h[:, 2] / props_h[:, 2])
        dh_h = wh * torch.log(target_h[:, 3] / props_h[:, 3])

        dx_o = wx * (target_o[:, 0] - props_o[:, 0])
        dy_o = wy * (target_o[:, 1] - props_o[:, 1])
        dw_o = ww * torch.log(target_o[:, 2] / props_o[:, 2])
        dh_o = wh * torch.log(target_o[:, 3] / props_o[:, 3])

        box_deltas = torch.stack([dx_h, dy_h, dw_h, dh_h, dx_o, dy_o, dw_o, dh_o], dim=1)

        return box_deltas

    def decode(self, props_h: Tensor, props_o: Tensor, box_deltas: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Recover the regressed box pairs based on the proposed pairs and the box deltas.
        NOTE that the proposed box pairs are presumed to have been normalised by image
        width and height and are in (c_x, c_y, w, h) format.

        Parameters:
        -----------
        props_h: Tensor
            (N, 4) Human box proposals
        props_o: Tensor
            (N, 4) Object box proposals
        box_deltas: Tensor
            (N, 8) Predicted regression values for proposed box pairs

        Returns:
        --------
        regressed_h: Tensor
            (N, 4) Regressed human boxes
        regressed_o: Tensor
            (N, 4) Regressed object boxes
        """
        weights = torch.as_tensor(self.weights).repeat(2).to(box_deltas)
        box_deltas = box_deltas / weights

        dx_h, dy_h, dw_h, dh_h, dx_o, dy_o, dw_o, dh_o = box_deltas.unbind(1)

        # # Prevent sending too large values into torch.exp()
        dw_h = torch.clamp(dw_h, max=self.bbox_xform_clip)
        dh_h = torch.clamp(dh_h, max=self.bbox_xform_clip)
        dw_o = torch.clamp(dw_o, max=self.bbox_xform_clip)
        dh_o = torch.clamp(dh_o, max=self.bbox_xform_clip)

        regressed_h = torch.stack([
            props_h[:, 0] + dx_h, props_h[:, 1] + dy_h,
            props_h[:, 2] * torch.exp(dw_h), props_h[:, 3] * torch.exp(dh_h)
        ], dim=1)

        regressed_o = torch.stack([
            props_o[:, 0] + dx_o, props_o[:, 1] + dy_o,
            props_o[:, 2] * torch.exp(dw_o), props_o[:, 3] * torch.exp(dh_o)
        ], dim=1)

        return regressed_h, regressed_o

class HungarianMatcher(nn.Module):
    def __init__(self,
        cost_object: float = 1., cost_verb: float = 1.,
        cost_bbox: float = 1., cost_giou: float = 1.
    ) -> None:
        """
        Parameters:
        ----------
        cost_object: float
            Weight on the object classification term
        cost_verb: float
            Weight on the verb classification term
        cost_bbox:
            Weight on the L1 regression error
        cost_giou:
            Weight on the GIoU term
        """
        super().__init__()
        self.cost_object = cost_object
        self.cost_verb = cost_verb
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_object + cost_verb + cost_bbox + cost_giou, \
            "At least one cost coefficient should be non zero."

    @torch.no_grad()
    def forward(self,
        bx_h: List[Tensor], bx_o: List[Tensor], objects: List[Tensor],
        prior: List[Tensor], logits: Tensor, targets: List[dict]
    ) -> List[Tensor]:
        """
        Parameters:
        ----------
        bh: List[Tensor]
            (M, 4) Human bounding boxes in detected pairs
        bo: List[Tensor]
            (M, 4) Object bounding boxes in detected pairs
        objects: List[Tensor]
            (M,) Object class indices in each pair 
        prior: List[Tensor]
            (2, M, K) Object detection scores for the human and object boxes in each pair
        logits: Tensor
            (M_, K) Classification logits for all boxes pairs
        targets: List[dict]
            Targets for each image with the following keys, `boxes_h` (G, 4), `boxes_o` (G, 4),
            `labels` (G, 117), `objects` (G,)

        Returns:
        --------
        List[Tensor]
            A list of tuples for matched indices between detected pairs and ground truth pairs.

        """
        eps = 1e-6

        # The number of box pairs in each image
        n = [len(p) for p in bx_h]

        gt_bx_h = [t['boxes_h'] for t in targets]
        gt_bx_o = [t['boxes_o'] for t in targets]

        scores = [
            torch.sigmoid(lg) * p.prod(0)
            for lg, p in zip(logits.split(n), prior)
        ]
        gt_labels = [t['labels'] for t in targets]

        cost_verb = [
            -0.5 * (
                s.matmul(l.T) / (l.sum(dim=1).unsqueeze(0) + eps) +
                (1-s).matmul(1 - l.T) / (torch.sum(1 - l, dim=1).unsqueeze(0) + eps)
            ) for s, l in zip(scores, gt_labels)
        ]

        cost_bbox = [torch.max(
            torch.cdist(h, gt_h, p=1), torch.cdist(o, gt_o, p=1)
        ) for h, o, gt_h, gt_o in zip(bx_h, bx_o, gt_bx_h, gt_bx_o)]

        cost_giou = [torch.max(
            -generalized_box_iou(box_cxcywh_to_xyxy(h), box_cxcywh_to_xyxy(gt_h)),
            -generalized_box_iou(box_cxcywh_to_xyxy(o), box_cxcywh_to_xyxy(gt_o))
        ) for h, o, gt_h, gt_o in zip(bx_h, bx_o, gt_bx_h, gt_bx_o)]

        cost_object = [
            -torch.log(                                 # Log barrier
                obj.unsqueeze(1).eq(t['object'])        # Binary mask
                * p[0].max(-1)[0].unsqueeze(1) + eps    # Object classification score
            ) for obj, p, t in zip(objects, prior, targets)
        ]

        # Final cost matrix
        C = [
            c_v * self.cost_verb + c_b * self.cost_bbox +
            c_g * self.cost_giou + c_o * self.cost_object
            for c_v, c_b, c_g, c_o in zip(cost_verb, cost_bbox, cost_giou, cost_object)
        ]

        indices = [linear_sum_assignment(c.cpu()) for c in C]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.matcher = HungarianMatcher(
            cost_object=args.set_cost_object,
            cost_verb=args.set_cost_verb,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou
        )
        self.box_pair_coder = BoxPairCoder()

    def focal_loss(self,
        bx_h: List[Tensor], bx_o: List[Tensor], indices: List[Tensor],
        prior: List[Tensor], logits: Tensor, targets: List[dict]
    ) -> Tensor:
        collated_labels = []
        for bh, bo, idx, tgt in zip(bx_h, bx_o, indices, targets):
            idx_h, idx_o = idx

            mask = torch.diag(torch.min(
                box_ops.box_iou(
                    box_cxcywh_to_xyxy(bh[idx_h]),
                    box_cxcywh_to_xyxy(tgt['boxes_h'][idx_o])
                ), box_ops.box_iou(
                    box_cxcywh_to_xyxy(bo[idx_h]),
                    box_cxcywh_to_xyxy(tgt['boxes_o'][idx_o])
                )
            ) > 0.5).unsqueeze(1)
            matched_labels = tgt['labels'][idx_o] * mask
            labels = torch.zeros(
                len(bh), self.args.num_classes,
                device=matched_labels.device
            )
            labels[idx_h] = matched_labels
            collated_labels.append(labels)

        collated_labels = torch.cat(collated_labels)
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = logits[x, y]; prior = prior[x, y]; labels = collated_labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                (prior + 1e-8) / (1 + torch.exp(-logits) - prior)
            ), labels, reduction='sum', alpha=self.args.alpha, gamma=self.args.gamma
        )
        return loss / n_p

    def regression_loss(self,
        props_h: List[Tensor], props_o: List[Tensor],
        reg_h: List[Tensor], reg_o: List[Tensor], indices: List[Tensor],
        targets: List[dict], bbox_deltas: List[Tensor],
    ) -> Tensor:
        props_h = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, props_h)])
        props_o = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, props_o)])
        reg_h = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, reg_h)])
        reg_o = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, reg_o)])

        tgt_h = torch.cat([t['boxes_h'][j].view(-1, 4) for (_, j), t in zip(indices, targets)])
        tgt_o = torch.cat([t['boxes_o'][j].view(-1, 4) for (_, j), t in zip(indices, targets)])

        bbox_deltas = torch.cat([d[i].view(-1, 8) for (i, _), d in zip(indices, bbox_deltas)])
        reg_targets = self.box_pair_coder.encode(
            props_h, props_o, tgt_h, tgt_o
        )

        huber_loss = F.smooth_l1_loss(
            bbox_deltas, reg_targets,
            beta=1 / 9, reduction='sum'
        )
        huber_loss = huber_loss / len(bbox_deltas)

        giou_loss = 2 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(reg_h),
            box_cxcywh_to_xyxy(tgt_h)
        )) - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(reg_o),
            box_cxcywh_to_xyxy(tgt_o)
        ))

        giou_loss = giou_loss.sum() / len(bbox_deltas)

        return dict(huber_loss=huber_loss, giou_loss=giou_loss)


    def forward(self,
        boxes: List[Tensor], bh: List[Tensor], bo: List[Tensor], objects: List[Tensor],
        prior: List[Tensor], logits: Tensor, bbox_deltas: Tensor, targets: List[dict]
    ) -> Dict[str, Tensor]:
        # n = [len(b) for b in bh]

        bx_h = [b[h] for b, h in zip(boxes, bh)]
        bx_o = [b[o] for b, o in zip(boxes, bo)]

        # bx_h_post, bx_o_post = self.box_pair_coder.decode(torch.cat(bx_h), torch.cat(bx_o), bbox_deltas)
        # bx_h_post = bx_h_post.split(n); bx_o_post = bx_o_post.split(n)

        indices = self.matcher(bx_h, bx_o, objects, prior, logits, targets)

        loss_dict = {"focal_loss": self.focal_loss(bx_h, bx_o, indices, prior, logits, targets)}
        # loss_dict.update(self.regression_loss(
        #     bx_h, bx_o, bx_h_post, bx_o_post, indices, targets, bbox_deltas.split(n)
        # ))

        return loss_dict

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def compute_spatial_encodings(
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
) -> Tensor:
    """
    Parameters:
    -----------
    boxes_1: List[Tensor]
        First set of bounding boxes (M, 4)
    boxes_1: List[Tensor]
        Second set of bounding boxes (M, 4)
    shapes: List[Tuple[int, int]]
        Image shapes, heights followed by widths
    eps: float
        A small constant used for numerical stability

    Returns:
    --------
    Tensor
        Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape

        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)

def binary_focal_loss_with_logits(
    x: Tensor, y: Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = 'mean',
    eps: float = 1e-6
) -> Tensor:
    """
    Focal loss by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf

    L = - |1-y-alpha| * |y-x|^{gamma} * log(|1-y-x|)

    Parameters:
    -----------
    x: Tensor[N, K]
        Post-normalisation scores
    y: Tensor[N, K]
        Binary labels
    alpha: float
        Hyper-parameter that balances between postive and negative examples
    gamma: float
        Hyper-paramter suppresses well-classified examples
    reduction: str
        Reduction methods
    eps: float
        A small constant to avoid NaN values from 'PowBackward'

    Returns:
    --------
    loss: Tensor
        Computed loss tensor
    """
    loss = (1 - y - alpha).abs() * ((y-torch.sigmoid(x)).abs() + eps) ** gamma * \
        torch.nn.functional.binary_cross_entropy_with_logits(
            x, y, reduction='none'
        )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unsupported reduction method {}".format(reduction))

def normalize_boxes(boxes, image_size, patch_size):
    """
    Normalize bounding box coordinates to match the ViT Patch Grid scale.
    
    Args:
        boxes (list of Tensors): Each element is a (num_objects_i, 4) tensor of (x_min, y_min, x_max, y_max).
        image_size (tuple): (H, W) -> Original image dimensions (e.g., 224x224).
        patch_size (int): Size of the ViT patches (e.g., 16 for ViT-B/16).

    Returns:
        list of Tensors: Normalized bounding boxes per image.
    """
    H, W = image_size
    N_h, N_w = H // patch_size, W // patch_size  # Compute patch grid size
    
    normalized_boxes = []
    for box in boxes:
        box[:, [0, 2]] = (box[:, [0, 2]] / W) * N_w  # Scale x_min, x_max
        box[:, [1, 3]] = (box[:, [1, 3]] / H) * N_h  # Scale y_min, y_max
        normalized_boxes.append(box)
    
    return normalized_boxes

def compute_box_distance(boxes_sub, boxes_obj, patch_size, image_size, sigma=1.5):
    """
    Compute normalized distances between bounding box centers and ViT patches.

    Args:
        boxes (list of Tensors): List where each tensor is (num_objects_i, 4) with (x_min, y_min, x_max, y_max).
        patch_grid (Tensor): (batch, num_patches, 2) -> (x, y) coordinates of each patch.
        image_size (tuple): (H, W) -> Original image dimensions.
        patch_size (int): Size of the ViT patches.

    Returns:
        list of Tensors: Each tensor is (num_objects_i, num_patches) containing normalized box distances.
    """
    # Normalize box coordinates to match the Patch Grid scale
    # boxes = normalize_boxes(boxes, image_size, patch_size)

    box_distances = []
    for i, (box_s, box_o) in enumerate(zip(boxes_sub, boxes_obj)):
        # 初始化 mask (batch-wise)
        device = box_s.device
        scale_factor = patch_size / image_size  # 将 bounding box 映射到 14x14 grid

        # 将 bounding box 转换到 patch grid
        box_s = (box_s * scale_factor).long().to(device)
        box_o = (box_o * scale_factor).long().to(device)

        mask = torch.zeros((len(box_s), patch_size, patch_size), dtype=torch.float32, device=box_s.device)

        # 生成 human, object, 和 union box mask
        x_min = torch.minimum(box_s[:, 0], box_o[:, 0])
        y_min = torch.minimum(box_s[:, 1], box_o[:, 1])
        x_max = torch.maximum(box_s[:, 2], box_o[:, 2])
        y_max = torch.maximum(box_s[:, 3], box_o[:, 3])

        # 计算 batch-wise mask
        grid_x, grid_y = torch.meshgrid(torch.arange(patch_size, device=box_s.device), torch.arange(patch_size, device=box_s.device), indexing='ij')

        human_mask = ((grid_x[None, :, :] >= box_s[:, None, None, 0]) & 
                    (grid_x[None, :, :] <= box_s[:, None, None, 2]) & 
                    (grid_y[None, :, :] >= box_s[:, None, None, 1]) & 
                    (grid_y[None, :, :] <= box_s[:, None, None, 3])).float() 

        object_mask = ((grid_x[None, :, :] >= box_o[:, None, None, 0]) & 
                    (grid_x[None, :, :] <= box_o[:, None, None, 2]) & 
                    (grid_y[None, :, :] >= box_o[:, None, None, 1]) & 
                    (grid_y[None, :, :] <= box_o[:, None, None, 3])).float() 

        union_mask = ((grid_x[None, :, :] >= x_min[:, None, None]) & 
                    (grid_x[None, :, :] <= x_max[:, None, None]) & 
                    (grid_y[None, :, :] >= y_min[:, None, None]) & 
                    (grid_y[None, :, :] <= y_max[:, None, None])).float() 

        # 计算 final mask
        mask = torch.zeros_like(human_mask).to(human_mask.device)
        reversed_mask = 1 - union_mask
        mask[reversed_mask>0] = -1E9

        # **使用 2D 卷积进行高斯平滑**
        kernel_size = 3  # 控制平滑范围
        sigma = sigma
        kernel = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # 2D 高斯核
        gaussian_kernel = kernel[:, None] * kernel[None, :]
        gaussian_kernel = gaussian_kernel.expand(1, 1, kernel_size, kernel_size)

        # 进行 2D 高斯平滑
        mask = mask.unsqueeze(1)  # (B, 1, 14, 14)
        pad_size = kernel_size // 2
        mask = F.conv2d(F.pad(mask, (pad_size, pad_size, pad_size, pad_size), mode='replicate'), gaussian_kernel, groups=1).squeeze(1)
        box_distances.append(mask.view(-1, patch_size * patch_size))

    return box_distances
    
def compute_attention_mask(box_distances, lambda_factor=10):
    """
    计算 attention mask,用高斯核衰减远离目标的 attention。
    """
    return torch.exp(-lambda_factor * box_distances)

def generate_patch_grid(image_size, patch_size, batch_size):
    """
    生成 CLIP ViT 的 Patch Grid 位置
    :param image_size: 输入图像的尺寸 (H, W)
    :param patch_size: ViT 处理的 Patch 尺寸 (P, P)
    :param batch_size: batch 大小
    :return: Tensor (batch, num_patches, 2) -> 每个 patch 的中心坐标
    """
    H, W = image_size
    P = patch_size
    N_h, N_w = H // P, W // P  # 计算网格大小

    # 生成 Patch 的 x, y 坐标索引 (网格)
    grid_x = torch.arange(N_w).float() * P + P / 2  # 水平方向
    grid_y = torch.arange(N_h).float() * P + P / 2  # 垂直方向
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='xy')  # 生成网格

    # 合并成 (num_patches, 2) 形状
    patch_grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (num_patches, 2)
    
    # 复制 batch 维度
    patch_grid = patch_grid.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_patches, 2)
    
    return patch_grid


    """
    Pad object prompts so that all images in the batch have the same number of objects.
    
    Args:
        object_prompts (list of Tensors): Each tensor is (num_objects_i, feature_dim).
        max_objects (int): The maximum number of objects in any image in the batch.
    
    Returns:
        Tensor: Padded object prompts of shape (batch, max_objects, feature_dim).
        Tensor: Mask indicating valid objects (1 for valid, 0 for padding).
    """
    # Pad object prompts to the same number of objects
    padded_prompts = pad_sequence(object_prompts, batch_first=True)  # (batch, max_objects, feature_dim)
    
    # Create a mask: 1 for valid objects, 0 for padded ones
    object_mask = torch.zeros((len(object_prompts), max_objects), dtype=torch.bool)
    for i, prompt in enumerate(object_prompts):
        object_mask[i, :prompt.size(0)] = 1  # Mark valid objects
    
    return padded_prompts, object_mask
