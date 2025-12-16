#【核心入口】 训练的主程序。
# 负责参数解析、构建模型、加载数据、训练循环（Epoch loop）和验证。
import argparse
import numpy as np
import os
import sys
sys.path.append('detr')
import torch
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.ops.boxes import box_iou
import hico_text_label
from hico_text_label import hico_unseen_index
from hola_model import build_detector
from utils_tip_cache_and_union_finetune import custom_collate, CustomisedDLE, DataFactory
import warnings
warnings.filterwarnings("ignore")

#计算一个数据集中不同的人-物交互(Human-Object Interaction, HOI)类别之间的共现(co-occurrence)概率矩阵
def tranverse_and_get_hoi_cooccurence(dataset):
    category = dataset.num_interation_cls
    hoi_cooccurence = torch.zeros(category, category)
    #遍历数据集中的每一张图像的标注 (anno in dataset.)
    for anno in dataset._anno:
        num_gt = len(anno['hoi'])
        for i in range(num_gt):
            for j in range(i+1, num_gt):
                #计算第i个 HOI 标注和第j个 HOI 标注中，人框 (boxes_h) 和物框 (boxes_o) 之间的 IoU
                h_iou = box_iou(torch.as_tensor(anno['boxes_h'][i:i+1]), torch.as_tensor(anno['boxes_h'][j:j+1]))
                o_iou = box_iou(torch.as_tensor(anno['boxes_o'][i:i+1]), torch.as_tensor(anno['boxes_o'][j:j+1]))
                if min(h_iou.item(), o_iou.item()) > 0.5:
                    #如果两个高度重叠的标注的 HOI 类别也相同，则跳过
                    if anno['hoi'][i] == anno['hoi'][j]:
                        continue
                    hoi_cooccurence[anno['hoi'][i],anno['hoi'][j]] += 1
                    hoi_cooccurence[anno['hoi'][j],anno['hoi'][i]] += 1
    #归一化为条件概率
    hoi_cooccurence = hoi_cooccurence.t() / (hoi_cooccurence.sum(dim=-1) + 1e-9)
    hoi_cooccurence = hoi_cooccurence.t()
    return hoi_cooccurence

#生成一个HICO-DET 数据集中人-物交互 (HOI) 类别到其对应的动词 (Verb) 类别和物体 (Object) 类别的索引对应关系列表
def hico_class_corr():
    """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]
        Returns:
            list[list[3]]
        """
    class_corr = []
    #这里的hico_text_label是字典，包含如'a photo of a person boarding an airplane'
    for i, (k, v) in enumerate(hico_text_label.hico_text_label.items()):
        class_corr.append([i, k[1], k[0]])
    return class_corr

def main(rank, args):
    # 初始化分布式训练环境
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )
    from torch.utils.tensorboard import SummaryWriter
    def is_main_process():
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    writer = None
    if is_main_process():
        tb_dir = os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    # 设置随机种子以确保实验的可重复性
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'

    elif args.clip_model_name == 'ViT-B-32':
        args.clip_model_name = 'ViT-B/32'

    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'

    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root, 
                           clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type, 
                           num_classes=args.num_classes, syn = None)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root, 
                          clip_model_name=args.clip_model_name) 
    verb2interaction = None
    # trainset[0][1]: dict_keys(['boxes_h', 'boxes_o', 'hoi', 'object', 'verb', 'orig_size', 'labels', 'size', 'filename'])
    # trainset[0][0]: (torch.Size([3, 814, 640]), torch.Size([3, 224, 224]))
    #训练集子集划分
    if args.training_set_ratio < 0.9:
        print(f'[INFO]: using {args.training_set_ratio} trainset to train!')
        sub_trainset, valset = trainset.dataset.split(args.training_set_ratio)
        trainset.dataset = sub_trainset
        trainset.keep = [i for i in range(len(sub_trainset))]
    #分布式数据加载器
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        sampler=DistributedSampler(trainset, num_replicas=args.world_size, rank=rank)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )

    args.human_idx = 0
    object_n_verb_to_interaction = train_loader.dataset.dataset.object_n_verb_to_interaction

    if args.dataset == 'hicodet':
        if args.num_classes == 117:
            #预测动词类别
            object_to_target = train_loader.dataset.dataset.object_to_verb
        elif args.num_classes == 600:
            #预测人-物交互类别
            object_to_target = train_loader.dataset.dataset.object_to_interaction
        if args.zs:
            #预测零样本HOI类别：如果 "骑马" 是未见类别，模型需要利用已见的 "骑自行车" 和 "看马" 等知识进行推理
            object_to_target = train_loader.dataset.zs_object_to_target
    print('[INFO]: num_classes', args.num_classes)

    #获取稀有/非稀有标注数量
    num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
    if args.num_classes == 117:
        num_anno = torch.as_tensor(trainset.dataset.anno_action)
    #构建检测器
    upt = build_detector(args, object_to_target, object_n_verb_to_interaction=object_n_verb_to_interaction, clip_model_path=args.clip_dir_vit, num_anno=num_anno, verb2interaction=verb2interaction)
    #评估时改变目标映射
    if args.dataset == 'hicodet' and args.eval:
        if args.num_classes == 117:#预测动词类别
            upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_verb
        else:#预测人-物交互类别
            upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_interaction
    #权重加载与状态恢复
    if os.path.exists(args.resume_part):
        #部分加载权重
        print(f"===>>> Rank {rank}: partially continue from saved checkpoint {args.resume}")
        checkpoint_part = torch.load(args.resume_part, map_location='cpu')
        upt.load_state_dict(checkpoint_part['model_state_dict'], strict = False)#strict = False允许加载时跳过不匹配的层
    if os.path.exists(args.resume):
        #完整加载检查点
        print(f"===>>> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')  # 
        if 'e632da11' in args.pretrained and args.dataset == 'hicodet':
            model_dict = checkpoint['model_state_dict']
            model_dict = {k: v for k, v in model_dict.items() if 'detector.class_embed' not in k}
            upt.load_state_dict(model_dict, strict = False)
        else:
            upt.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        #随机初始化开始训练
        print(f"===>>> Rank {rank}: start from a randomly initialised model")
    #训练引擎初始化
    if os.path.exists(args.resume):
        #恢复训练
        engine = CustomisedDLE(
            upt,                                 # 模型实例
            train_loader,                        # 训练数据加载器
            max_norm=args.clip_max_norm,         # 梯度裁剪的最大范数
            num_classes=args.num_classes,        # 目标类别总数
            print_interval=args.print_interval,  # 训练日志打印间隔
            find_unused_parameters=True,         # 在分布式训练中启用，用于处理梯度计算中未使用的参数。
            cache_dir=args.output_dir,           # 缓存和输出目录
            start_epoch = checkpoint['epoch'],    # 从检查点恢复训练的起始epoch
            writer = writer
        )
    else:
        #从头开始训练
        engine = CustomisedDLE(
            upt, train_loader,
            max_norm=args.clip_max_norm,
            num_classes=args.num_classes,
            print_interval=args.print_interval,
            find_unused_parameters=True,
            cache_dir=args.output_dir,
            writer = writer
        )
    #logit尺度调整，通过因子vis_tor
    if args.vis_tor != 1 and (args.eval or args.cache):
        upt.logit_scale_HO = torch.nn.Parameter(upt.logit_scale_HO * args.vis_tor)
        upt.logit_scale_U = torch.nn.Parameter(upt.logit_scale_U * args.vis_tor)
    #缓存模式：仅运行前向传播，提取并保存中间特征（如CLIP特征、检测结果）
    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        return
    #评估模式
    if args.eval:
        upt.eval()
        ap = engine.test_hico(test_loader, args)
        # Fetch indices for rare and non-rare classes
        print("ap", ap)
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)#rare：标注数量少于10次的类别索引
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        #计算并打印整体、稀有类别和非稀有类别的平均精度均值 (mAP)
        print(
            f"The mAP is {ap.mean()*100:.2f},"
            f" rare: {ap[rare].mean()*100:.2f},"
            f" none-rare: {ap[non_rare].mean()*100:.2f},"
        )
        #零样本 (Zero-Shot) 性能分解
        if args.zs:
            zs_hoi_idx = hico_unseen_index[args.zs_type]
            print(f'>>> zero-shot setting({args.zs_type}!!)')
            ap_unseen = []
            ap_seen = []
            for i, value in enumerate(ap):
                if i in zs_hoi_idx: 
                    ap_unseen.append(value)
                else: 
                    ap_seen.append(value)
            ap_unseen = torch.as_tensor(ap_unseen).mean()
            ap_seen = torch.as_tensor(ap_seen).mean()
            #打印零样本设置下的性能分解
            print(
                f"full mAP: {ap.mean()*100:.2f}",
                f"unseen: {ap_unseen*100:.2f}",
                f"seen: {ap_seen*100:.2f}",
            )
        return
    #冻结基础检测器的主干参数
    for p in upt.detector.parameters():
        p.requires_grad = False
    for n, p in upt.clip_head.named_parameters():
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj'): 
            #解冻CLIP的特定组件
            p.requires_grad = True
            print(n)
        elif 'adaptermlp' in n or "prompt_learner" in n:
            #解冻Adapter/Prompt Learning组件
            p.requires_grad = True
        else: #其他CLIP Head中的参数（如大部分Transformer层）被冻结
            p.requires_grad = False

    if args.frozen_classifier != None:
        frozen_name_lst = []
        if 'HO' in args.frozen_classifier:
            frozen_name_lst.append('adapter_HO')
        if 'U' in args.frozen_classifier:
            frozen_name_lst.append('adapter_U')
        if 'T' in args.frozen_classifier:
            frozen_name_lst.append('adapter_union')
        
        for n, p in upt.named_parameters():
            #跳过detector和clip_head中已处理的参数
            if 'clip_head' in n or 'detector' in n:
                continue
            #冻结指定名称的Adapter（分类器）参数
            if n.split('.')[0] in frozen_name_lst:
                p.requires_grad = False
    
    if args.label_learning:
        #启用学习类别标签嵌入的参数
        for n, p in upt.named_parameters():
            if 'clip_head' in n or 'detector' in n:
                continue
            if 'label_' in n:
                p.requires_grad = True
    # others = [n for n, p in upt.named_parameters()
    #                 if p.requires_grad and 'clip_head' not in n]
    param_dicts = [
        #第一个组：CLIP Head中所有可训练的参数
        {"params": [p for n, p in upt.clip_head.named_parameters()if p.requires_grad]},
        #第二个组：模型中除了CLIP Head以外所有可训练的参数
        {"params": [p for n, p in upt.named_parameters()if p.requires_grad and 'clip_head' not in n],
         "lr": args.lr_head,},
    ]
    #统计可训练参数数量
    n_parameters = sum(p.numel() for p in upt.parameters() if p.requires_grad)
    print('number of leanable params:', n_parameters)
    #统计所有参数数量
    n_parameters = sum(p.numel() for p in upt.parameters())
    print('number of all params:', n_parameters)
    #优化器
    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_vit,
        weight_decay=args.weight_decay
    )
    #学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    #恢复优化器和调度器状态（如果从检查点恢复）
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch=checkpoint['epoch']
        iteration = checkpoint['iteration']
        #恢复混合精度训练的Scaler
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        #更新训练引擎的状态
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch,iteration=iteration, scaler=scaler)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    #保存训练参数
    import json
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()
    #启动
    engine(args.epochs)

#一个健康检查（sanity check）函数,验证数据正常 + 模型正常前向 + 接口对齐
@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.human_idx = 0; args.num_classes = 117
    object_to_target = dataset.dataset.object_to_verb
    upt = build_detector(args, object_to_target)
    if args.eval:
        upt.eval()

    image, target = dataset[0]
    outputs = upt([image], [target])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-3, type=float)#HOI学习率
    parser.add_argument('--lr-vit', default=1e-3, type=float)#vit学习率
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)#第几个 epoch 进行 LR decay
    parser.add_argument('--clip-max-norm', default=0.1, type=float)#防止梯度爆炸
    #DETR配置
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)#预测100个box
    parser.add_argument('--pre-norm', action='store_true')
    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,help="Relative classification weight of the no-object class")#no-object类别权重
    #HOI动作分类损失（focal loss）
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)
    #数据集
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')
    #训练参数
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--port', default='1233', type=str)#干嘛的？
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')#特征缓存
    parser.add_argument('--sanity', action='store_true')#只跑sanity_check
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)
    #CLIP ViT
    parser.add_argument('--visual_mode', default='vit', type=str)
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)

    # ---ViT-L/14@336px START: emb_dim: 768 
    parser.add_argument('--clip_visual_layers_vit', default=24, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=768, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=336, type=int)
    parser.add_argument('--clip_visual_width_vit', default=1024, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=14, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=768, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=12, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-L/14@336px----END----

    parser.add_argument('--clip_text_context_length_vit', default=77, type=int) # 13 -77
    parser.add_argument('--use_insadapter', action='store_true')
    
    parser.add_argument('--use_mean', action='store_true') # 13 -77
    parser.add_argument('--logits_type', default='HO+U+T', type=str) # 13 -77 # text_add_visual, visual
    parser.add_argument('--num_shot', default='2', type=int) # 13 -77 # text_add_visual, visual
    parser.add_argument('--file1', default='./hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p',type=str)
    parser.add_argument('--prior_type', type=str, default='cbe', choices=['cbe', 'cb', 'ce', 'be', 'c', 'b', 'e'])
    parser.add_argument('--training_set_ratio', type=float, default=1.0)
    parser.add_argument('--frozen_classifier', type=str, default=None)
    parser.add_argument('--zs', action='store_true') ## zero-shot
    parser.add_argument('--hyper_lambda', type=float, default=2.8)
    parser.add_argument('--use_weight_pred', action='store_true')
    parser.add_argument('--zs_type', type=str, default='rare_first', choices=['rare_first', 'non_rare_first', 'unseen_verb', 'unseen_object', 'uc0', 'uc1', 'uc2', 'uc3', 'uc4'])
    parser.add_argument('--vis_tor', type=float, default=1.0)
    parser.add_argument('--adapter_num_layers', type=int, default=1)

    # prompt learning
    parser.add_argument('--N_CTX', type=int, default=24)  # number of context vectors
    parser.add_argument('--CSC', type=bool, default=False)  # class-specific context
    parser.add_argument('--CTX_INIT', type=str, default='')  # initialization words
    parser.add_argument('--CLASS_TOKEN_POSITION', type=str, default='end')  # # 'middle' or 'end' or 'front'

    parser.add_argument('--use_templates', action='store_true') 
    parser.add_argument('--feat_mask_type', type=int, default=0,) # 0: dropout(random mask); 1: None
    parser.add_argument('--num_classes', type=int, default=117,) 
    parser.add_argument('--prior_method', type=int, default=0) ## 0: instance-wise, 1: pair-wise, 2: learnable
    parser.add_argument('--vis_prompt_num', type=int, default=50) ##  (prior_method == learnable)
    parser.add_argument('--box_proj', type=int, default=0,) ## 0: None; 1: f_u = ROI-feat + MLP(uni-box)
    parser.add_argument('--adapter_pos', type=str, default='all', choices=['all', 'front', 'end', 'random', 'last'])
    parser.add_argument('--use_multi_hot', action='store_true')
    parser.add_argument('--label_learning', action='store_true')
    parser.add_argument('--label_choice', default='random', choices=['random', 'single_first', 'multi_first', 'single+multi', 'rare_first', 'non_rare_first', 'rare+non_rare'])  
    parser.add_argument('--repeat_factor_sampling', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='apply repeat factor sampling to increase the rate at which tail categories are observed')
    
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--vis_img', action="store_true")
    parser.add_argument('--vis_img_path', default = "pred_annotation", type=str)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--resume_part', default='', help='Resume from a model')
    parser.add_argument('--fix_mem', default=False, action='store_true')
    parser.add_argument('--llmtxt', default=False, action='store_true')
    parser.add_argument('--img_align', default=False, action='store_true')
    parser.add_argument('--semloss_weight', type=int, default = 150)
    parser.add_argument('--self_adapt', default=False, action='store_true')
    parser.add_argument('--norm_pred', default=False, action='store_true')
    parser.add_argument('--wo_unseen_pred', default=False, action='store_true')

    ##### decomposition parameters #####
    parser.add_argument('--basis_feat_enable', default=False, action='store_true')
    parser.add_argument('--seperate_ho', default=0, type = int)
    parser.add_argument('--basis_feat_init', default='random', type=str, choices=('random', 'pca', 'co_pca'))
    parser.add_argument('--unique_basis_weights', default=False, action='store_true')
    parser.add_argument('--disentangle_basis', default=False, action='store_true')
    parser.add_argument('--basis_num', type=int, default = 100)
    parser.add_argument('--ao_sep_basis', default=False, action='store_true')
    parser.add_argument('--act_txtdecrip', default=False, action='store_true')
    parser.add_argument('--sep_frac', type=int, default = 3)
    parser.add_argument('--basis_constraint', default='quadratic', type=str, choices=('quadratic', 'direct'))
    parser.add_argument('--basis_feat_constraint', default='none', type=str, choices=('l2', 'kl', 'none'))
    parser.add_argument('--fix_act_w', default=False, action='store_true')  ### when calculate KL constraint from action weights to HOI weights
    parser.add_argument('--HOI_train_w_b', default='w', type=str, choices=('w', 'b', 'both', 'none'))  
    parser.add_argument('--no_act_constraint', default=False, action='store_true')
    parser.add_argument('--kl_t', type=float, default = 0.1)
    parser.add_argument('--recon_ratio_pca', type=float, default = 0.95)
    parser.add_argument('--wo_sparsity', default=False, action='store_true')
    parser.add_argument('--pt_learn', default=0, type=int)
    parser.add_argument('--pt_lyr', default=[1,9], type=list)
    parser.add_argument('--semloss', default=False, action='store_true')
    #### human-object tokens
    parser.add_argument('--ho_pair_pt', default=False, action='store_true')
    parser.add_argument('--ho_pair_prior', default=0, type=int)
    parser.add_argument('--pt_init', default='pos+detr+fus', choices=['pos', 'detr', 'pos+detr', 'pos+detr+fus'])
    parser.add_argument('--pred_type', type=str, default='ho+u', choices=['ho', 'u', 'l','ho+u', 'ho+l', 'u+l', 'ho+u+l'])
    parser.add_argument('--pt_attn',  type=str, default='uniform', choices=['mask', 'uniform'])
    args = parser.parse_args()

    if args.sanity:
        #只跑检查，不训练
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    if args.world_size==1:
        main(0,args)
    else:
        mp.spawn(main, nprocs=args.world_size, args=(args,))
