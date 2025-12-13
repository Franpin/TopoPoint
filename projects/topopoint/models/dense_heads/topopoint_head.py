import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss, build_head
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.topopoint.models.dense_heads.relationship_head import MLP

total_distance = {}

@HEADS.register_module()
class TopoPointHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_pts_query=100,
                 transformer=None,
                 lclc_head=None,
                 lcte_head=None,
                 bbox_coder=None,
                 num_reg_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 pc_range=None,
                 pts_dim =3,
                 sync_cls_avg_factor=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'

            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_pts_query = num_pts_query
        self.pts_dim = pts_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        
        if lclc_head is not None:
            self.lclc_cfg = lclc_head

        if lcte_head is not None:
            self.lcte_cfg = lcte_head

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 6
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.gt_c_save = self.code_size

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_reg_fcs = num_reg_fcs
        self._init_layers()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        pts_cls_branch = []
        for _ in range(self.num_reg_fcs):
            pts_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            pts_cls_branch.append(nn.LayerNorm(self.embed_dims))
            pts_cls_branch.append(nn.ReLU(inplace=True))
        pts_cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        pts_fc_cls = nn.Sequential(*pts_cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        pt_reg_branch = []
        for _ in range(self.num_reg_fcs):
            pt_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            pt_reg_branch.append(nn.ReLU())
        pt_reg_branch.append(Linear(self.embed_dims, 3))
        pt_reg_branch = nn.Sequential(*pt_reg_branch)

        ptlc_branch = build_head(self.lclc_cfg)
        lclc_branch = build_head(self.lclc_cfg)
        lcte_branch = build_head(self.lcte_cfg)

        te_embed_branch = []
        in_channels = self.embed_dims
        for _ in range(self.num_reg_fcs - 1):
            te_embed_branch.append(nn.Sequential(
                    Linear(in_channels, 2 * self.embed_dims),
                    nn.ReLU(),
                    nn.Dropout(0.1)))
            in_channels = 2 * self.embed_dims
        te_embed_branch.append(Linear(2 * self.embed_dims, self.embed_dims))
        te_embed_branch = nn.Sequential(*te_embed_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # mlp_branch = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 3)

        num_pred = self.transformer.decoder.num_layers
        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.pts_cls_branches = _get_clones(pts_fc_cls, num_pred)
        
        self.reg_branches = _get_clones(reg_branch, num_pred)
        self.pt_reg_branches = _get_clones(pt_reg_branch, num_pred)

        self.ptlc_branches = _get_clones(ptlc_branch, num_pred)
        self.lclc_branches = _get_clones(lclc_branch, num_pred)
        self.lcte_branches = _get_clones(lcte_branch, num_pred)

        self.te_embed_branches = _get_clones(te_embed_branch, num_pred)

        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

        # self.mlp_init = MLP(self.embed_dims*2, self.embed_dims*2, self.embed_dims*2, 3)
        # self.mlp_branches = _get_clones(mlp_branch, num_pred)
        self.pts_query_embedding = nn.Embedding(self.num_pts_query, self.embed_dims * 2)

    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
            for m in self.pts_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, bev_feats, img_metas, te_feats, te_cls_scores):
        

        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)

        # pts_object_query_embeds = self.mlp_init(object_query_embeds.clone().detach())
        pts_object_query_embeds = self.pts_query_embedding.weight.to(dtype)
        te_feats = torch.stack([self.te_embed_branches[lid](te_feats[lid]) for lid in range(len(te_feats))])

        outputs = self.transformer(
            mlvl_feats,
            bev_feats,
            object_query_embeds,
            pts_object_query_embeds,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            pts_cls_branches=self.pts_cls_branches,
            cls_branches = self.cls_branches,
            ptlc_branches=self.ptlc_branches,
            # mlp_branches=self.mlp_branches,
            mlp_branches=None,
            lclc_branches=self.lclc_branches,
            lcte_branches=self.lcte_branches,
            reg_branches = self.reg_branches,
            pt_reg_branches = self.pt_reg_branches,
            te_feats=te_feats,
            te_cls_scores=te_cls_scores,
            img_metas=img_metas,
        )

        hs, pts_hs, init_reference, inter_references, ptlc_rel_out, lclc_rel_out, lcte_rel_out, lc_outputs_coords, pts_outputs_coords,outputs_classes,pts_outputs_classes = outputs

        outs = {
            'all_cls_scores': outputs_classes,
            'all_pts_cls_scores': pts_outputs_classes,
            'all_lanes_preds': lc_outputs_coords,
            'all_pts_preds': pts_outputs_coords,
            'all_ptlc_preds': ptlc_rel_out,
            'all_lclc_preds': lclc_rel_out,
            'all_lcte_preds': lcte_rel_out,
            'history_states': hs
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           lanes_pred,
                           pts_cls_score,
                           pts_pred,
                           ptlc_pred,
                           lclc_pred,
                           gt_labels,
                           gt_lanes,
                           gt_pts_labels,
                           gt_pts,
                           gt_ptlc_adj,
                           gt_lane_adj,
                           gt_bboxes_ignore=None):
        num_bboxes = lanes_pred.size(0)
        num_pts = pts_pred.size(0)
        # assigner and sampler

        pts_assign_result = self.assigner.assign(pts_pred, pts_cls_score, gt_pts,
                                             gt_pts_labels, gt_bboxes_ignore)

        pts_sampling_result = self.sampler.sample(pts_assign_result, pts_pred,
                                              gt_pts)

        gt_lanes = gt_lanes.reshape(gt_lanes.shape[0], -1)

        lane_assign_result = self.assigner.assign(lanes_pred, cls_score, gt_lanes,
                                             gt_labels, gt_bboxes_ignore)

        lane_sampling_result = self.sampler.sample(lane_assign_result, lanes_pred,
                                              gt_lanes)
        
        lane_pos_inds = lane_sampling_result.pos_inds
        lane_neg_inds = lane_sampling_result.neg_inds
        lane_pos_assigned_gt_inds = lane_sampling_result.pos_assigned_gt_inds

        lane_labels = gt_lanes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        lane_labels[lane_pos_inds] = gt_labels[lane_sampling_result.pos_assigned_gt_inds].long()
        lane_label_weights = gt_lanes.new_ones(num_bboxes)

        pts_pos_inds = pts_sampling_result.pos_inds
        pts_neg_inds = pts_sampling_result.neg_inds
        pts_pos_assigned_gt_inds = pts_sampling_result.pos_assigned_gt_inds

        pts_labels = gt_pts.new_full((num_pts,), self.num_classes, dtype=torch.long)
        pts_labels[pts_pos_inds] = gt_pts_labels[pts_sampling_result.pos_assigned_gt_inds].long()
        pts_label_weights = gt_pts.new_ones(num_pts)

        gt_c_pts = gt_pts.shape[-1]

        self.pts_gt_c_save = gt_c_pts

        # bbox targets
        gt_c = gt_lanes.shape[-1]
        if gt_c == 0:
            gt_c = self.gt_c_save
            lane_sampling_result.pos_gt_bboxes = torch.zeros((0, gt_c)).to(lane_sampling_result.pos_gt_bboxes.device)
        else:
            self.lane_gt_c_save = gt_c

        lane_bbox_targets = torch.zeros_like(lanes_pred)[..., :gt_c]
        lane_bbox_weights = torch.zeros_like(lanes_pred)
        lane_bbox_weights[lane_pos_inds] = 1.0

        pts_bbox_targets = torch.zeros_like(pts_pred)[..., :gt_c_pts]
        pts_bbox_weights = torch.zeros_like(pts_pred)
        pts_bbox_weights[pts_pos_inds] = 1.0
        # DETR
        pts_bbox_targets[pts_pos_inds] = pts_sampling_result.pos_gt_bboxes
        lane_bbox_targets[lane_pos_inds] = lane_sampling_result.pos_gt_bboxes

        lclc_target = torch.zeros_like(lclc_pred.squeeze(-1), dtype=gt_lane_adj.dtype, device=lclc_pred.device)
        xs = lane_pos_inds.unsqueeze(-1).repeat(1, lane_pos_inds.size(0))
        ys = lane_pos_inds.unsqueeze(0).repeat(lane_pos_inds.size(0), 1)
        # lclc_target[xs, ys] = gt_lane_adj[lane_pos_assigned_gt_inds][:, lane_pos_assigned_gt_inds]
        lclc_target = gt_lane_adj[lane_pos_assigned_gt_inds][:, lane_pos_assigned_gt_inds]
        lclc_pred = lclc_pred[xs, ys]


        ptlc_target = torch.zeros_like(ptlc_pred.squeeze(-1), dtype=gt_lane_adj.dtype, device=ptlc_pred.device)
        xs = pts_pos_inds.unsqueeze(-1).repeat(1, lane_pos_inds.size(0))
        ys = lane_pos_inds.unsqueeze(0).repeat(pts_pos_inds.size(0), 1)

        ptlc_target[xs, ys] = gt_ptlc_adj[pts_pos_assigned_gt_inds][:, lane_pos_assigned_gt_inds]
        # ptlc_target = gt_ptlc_adj[pts_pos_assigned_gt_inds][:, lane_pos_assigned_gt_inds]
        # ptlc_pred = ptlc_pred[xs, ys]

        return (pts_labels,lane_labels, pts_label_weights,lane_label_weights, pts_bbox_targets, lane_bbox_targets, ptlc_target, ptlc_pred, lclc_target, lclc_pred,
                pts_bbox_weights, lane_bbox_weights,
                pts_pos_inds, lane_pos_inds,pts_neg_inds,lane_neg_inds, 
                pts_pos_assigned_gt_inds,lane_pos_assigned_gt_inds)

    def get_targets(self,
                    cls_scores_list,
                    lanes_preds_list,
                    pts_cls_scores_list,
                    pts_preds_list,
                    ptlc_preds_list,
                    lclc_preds_list,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_end_pts_list,
                    gt_end_pts_label_list,
                    gt_ptlc_adj_list,
                    gt_lane_adj_list,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]


        (pts_labels,lane_labels, pts_label_weights,lane_label_weights, pts_bbox_targets, lane_bbox_targets, ptlc_targets, ptlc_preds, lclc_targets, lclc_preds,
                pts_bbox_weights, lane_bbox_weights,
                pts_pos_inds, lane_pos_inds,pts_neg_inds,lane_neg_inds, 
                pts_pos_assigned_gt_inds,lane_pos_assigned_gt_inds)= multi_apply(
            self._get_target_single, cls_scores_list, lanes_preds_list, pts_cls_scores_list, pts_preds_list, ptlc_preds_list, lclc_preds_list,
            gt_labels_list, gt_lanes_list, gt_end_pts_label_list, gt_end_pts_list, gt_ptlc_adj_list, gt_lane_adj_list, gt_bboxes_ignore_list)


        pts_num_total_pos = sum((inds.numel() for inds in pts_pos_inds))
        pts_num_total_neg = sum((inds.numel() for inds in pts_neg_inds))

        lane_num_total_pos = sum((inds.numel() for inds in lane_pos_inds))
        lane_num_total_neg = sum((inds.numel() for inds in lane_neg_inds))


        pts_assign_result = dict(
            pos_inds=pts_pos_inds, neg_inds=pts_neg_inds, pos_assigned_gt_inds=pts_pos_assigned_gt_inds
        )

        lane_assign_result = dict(
            pos_inds=lane_pos_inds, neg_inds=lane_neg_inds, pos_assigned_gt_inds=lane_pos_assigned_gt_inds
        )
        return (pts_labels, lane_labels,pts_label_weights,lane_label_weights, pts_bbox_targets,lane_bbox_targets, ptlc_targets, ptlc_preds, lclc_targets, lclc_preds,
                pts_bbox_weights,lane_bbox_weights, 
                pts_num_total_pos, pts_num_total_neg, lane_num_total_pos,lane_num_total_neg,
                pts_assign_result,lane_assign_result)

    def loss_single(self,
                    cls_scores,
                    lanes_preds,
                    pts_cls_scores,
                    pts_preds,
                    ptlc_preds,
                    lclc_preds,
                    lcte_preds,
                    te_assign_result,
                    gt_lanes_list,
                    gt_labels_list,
                    all_gt_end_pts_list, 
                    all_gt_end_pts_label_list,
                    gt_ptlc_adj_list,
                    gt_lane_adj_list,
                    gt_lane_lcte_adj_list,
                    layer_index,
                    gt_bboxes_ignore_list=None):



        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        lanes_preds_list = [lanes_preds[i] for i in range(num_imgs)]

        pts_cls_scores_list = [pts_cls_scores[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        ptlc_preds_list = [ptlc_preds[i] for i in range(num_imgs)]
        lclc_preds_list = [lclc_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, lanes_preds_list, pts_cls_scores_list, pts_preds_list, ptlc_preds_list, lclc_preds_list, 
                                           gt_lanes_list, gt_labels_list, all_gt_end_pts_list, all_gt_end_pts_label_list, 
                                           gt_ptlc_adj_list, gt_lane_adj_list, gt_bboxes_ignore_list)
                                           
         
        (pts_labels, lane_labels,pts_label_weights,lane_label_weights, pts_bbox_targets,lane_bbox_targets, ptlc_targets, ptlc_preds, lclc_targets, lclc_preds,
                pts_bbox_weights,lane_bbox_weights,
                pts_num_total_pos, pts_num_total_neg, lane_num_total_pos,lane_num_total_neg,
                pts_assign_result,lane_assign_result) = cls_reg_targets


        pts_labels = torch.cat(pts_labels, 0)
        pts_label_weights = torch.cat(pts_label_weights, 0)
        pts_bbox_targets = torch.cat(pts_bbox_targets, 0)
        pts_bbox_weights = torch.cat(pts_bbox_weights, 0)

        lane_labels = torch.cat(lane_labels, 0)
        lane_label_weights = torch.cat(lane_label_weights, 0)
        lane_bbox_targets = torch.cat(lane_bbox_targets, 0)
        lane_bbox_weights = torch.cat(lane_bbox_weights, 0)


        ptlc_targets = torch.cat(ptlc_targets, 0)
        ptlc_preds = torch.cat(ptlc_preds, 0)
        lclc_targets = torch.cat(lclc_targets, 0)
        lclc_preds = torch.cat(lclc_preds, 0)
        # classification loss
        pts_cls_scores = pts_cls_scores.reshape(-1, self.cls_out_channels)
        lane_cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo


        pts_cls_avg_factor = pts_num_total_pos * 1.0 + \
            pts_num_total_neg * self.bg_cls_weight
        lane_cls_avg_factor = lane_num_total_pos * 1.0 + \
            lane_num_total_neg * self.bg_cls_weight

        if self.sync_cls_avg_factor:
            pts_cls_avg_factor = reduce_mean(
                pts_cls_scores.new_tensor([pts_cls_avg_factor]))
        if self.sync_cls_avg_factor:
            lane_cls_avg_factor = reduce_mean(
                lane_cls_scores.new_tensor([lane_cls_avg_factor]))


        pts_cls_avg_factor = max(pts_cls_avg_factor, 1)
        lane_cls_avg_factor = max(lane_cls_avg_factor, 1)

        pts_loss_cls = self.loss_cls(
            pts_cls_scores, pts_labels, pts_label_weights, avg_factor=pts_cls_avg_factor)
        lane_loss_cls = self.loss_cls(
            lane_cls_scores, lane_labels, lane_label_weights, avg_factor=lane_cls_avg_factor)
        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        pts_num_total_pos = pts_loss_cls.new_tensor([pts_num_total_pos])
        pts_num_total_pos = torch.clamp(reduce_mean(pts_num_total_pos), min=1).item()
        lane_num_total_pos = lane_loss_cls.new_tensor([lane_num_total_pos])
        lane_num_total_pos = torch.clamp(reduce_mean(lane_num_total_pos), min=1).item()


        # regression L1 loss
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-1))
        isnotnan = torch.isfinite(pts_bbox_targets).all(dim=-1)
        pts_loss_bbox = self.loss_bbox(
            pts_preds[isnotnan, :3], 
            pts_bbox_targets[isnotnan, :3],
            pts_bbox_weights[isnotnan, :3],
            avg_factor=pts_num_total_pos)

        lanes_preds = lanes_preds.reshape(-1, lanes_preds.size(-1))
        isnotnan = torch.isfinite(lane_bbox_targets).all(dim=-1)

        lane_loss_bbox = self.loss_bbox(
            lanes_preds[isnotnan, :], 
            lane_bbox_targets[isnotnan, :],
            lane_bbox_weights[isnotnan, :],
            avg_factor=lane_num_total_pos)

        ptlc_targets = 1 - ptlc_targets.view(-1).long()
        ptlc_preds = ptlc_preds.view(-1, 1)
        loss_ptlc = self.ptlc_branches[layer_index].loss_rel(ptlc_preds, ptlc_targets)

        # lclc loss
        lclc_targets = 1 - lclc_targets.view(-1).long()
        lclc_preds = lclc_preds.view(-1, 1)
        loss_lclc = self.lclc_branches[layer_index].loss_rel(lclc_preds, lclc_targets)

        loss_lcte = self.lcte_branches[layer_index].loss(lcte_preds, gt_lane_lcte_adj_list, lane_assign_result, te_assign_result)['loss_rel']

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            pts_loss_cls = torch.nan_to_num(pts_loss_cls)
            pts_loss_bbox = torch.nan_to_num(pts_loss_bbox)
            lane_loss_cls = torch.nan_to_num(lane_loss_cls)
            lane_loss_bbox = torch.nan_to_num(lane_loss_bbox)
        return pts_loss_cls, pts_loss_bbox,lane_loss_cls, lane_loss_bbox, loss_ptlc, loss_lclc, loss_lcte

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             gt_lanes_3d,
             gt_labels_list,
             gt_end_pts,
             gt_end_pts_label_list,
             gt_ptlc_adj,
             gt_lane_adj,
             gt_lane_lcte_adj,
             te_assign_results,
             gt_bboxes_ignore=None,
             img_metas=None):

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_pts_cls_scores = preds_dicts['all_pts_cls_scores']
        all_pts_preds = preds_dicts['all_pts_preds']
        all_cls_scores = preds_dicts['all_cls_scores']
        all_lanes_preds = preds_dicts['all_lanes_preds']
        all_ptlc_preds = preds_dicts['all_ptlc_preds']
        all_lclc_preds = preds_dicts['all_lclc_preds']
        all_lcte_preds = preds_dicts['all_lcte_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_lanes_list = [lane for lane in gt_lanes_3d]
        gt_end_pts_list = [pts for pts in gt_end_pts]

        all_gt_lanes_list = [gt_lanes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_end_pts_list = [gt_end_pts_list for _ in range(num_dec_layers)]
        all_gt_end_pts_label_list = [gt_end_pts_label_list for _ in range(num_dec_layers)]

        all_gt_ptlc_adj_list = [gt_ptlc_adj for _ in range(num_dec_layers)]
        all_gt_lane_adj_list = [gt_lane_adj for _ in range(num_dec_layers)]
        all_gt_lane_lcte_adj_list = [gt_lane_lcte_adj for _ in range(num_dec_layers)]
        layer_index = [i for i in range(num_dec_layers)]

        pts_losses_cls, pts_losses_bbox, lane_losses_cls, lane_losses_bbox, losses_ptlc, losses_lclc, losses_lcte = multi_apply(
            self.loss_single, all_cls_scores, all_lanes_preds, all_pts_cls_scores, all_pts_preds, all_ptlc_preds, all_lclc_preds, all_lcte_preds, te_assign_results,
            all_gt_lanes_list, all_gt_labels_list, all_gt_end_pts_list, all_gt_end_pts_label_list, all_gt_ptlc_adj_list, all_gt_lane_adj_list, 
            all_gt_lane_lcte_adj_list, layer_index)

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_pts_cls'] = pts_losses_cls[-1]
        loss_dict['loss_pts_reg'] = pts_losses_bbox[-1]
        loss_dict['loss_lane_cls'] = lane_losses_cls[-1]
        loss_dict['loss_lane_reg'] = lane_losses_bbox[-1]
        loss_dict['loss_ptlc_rel'] = losses_ptlc[-1]
        loss_dict['loss_lclc_rel'] = losses_lclc[-1]
        loss_dict['loss_lcte_rel'] = losses_lcte[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for pts_loss_cls_i, pts_loss_bbox_i, lane_loss_cls_i, lane_loss_bbox_i, loss_ptlc_i, loss_lclc_i, loss_lcte_i in zip(
            pts_losses_cls[:-1], pts_losses_bbox[:-1], lane_losses_cls[:-1], lane_losses_bbox[:-1], losses_ptlc[:-1], losses_lclc[:-1], losses_lcte[:-1]):

            loss_dict[f'd{num_dec_layer}.loss_pts_cls'] = pts_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts_reg'] = pts_loss_bbox_i            
            loss_dict[f'd{num_dec_layer}.loss_lane_cls'] = lane_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_lane_reg'] = lane_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_ptlc_rel'] = loss_ptlc_i
            loss_dict[f'd{num_dec_layer}.loss_lclc_rel'] = loss_lclc_i
            loss_dict[f'd{num_dec_layer}.loss_lcte_rel'] = loss_lcte_i
            
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, rescale=False):

        num_query = preds_dicts['all_lanes_preds'].shape[2]

        lanes_preds = preds_dicts['all_lanes_preds'][-1]
        lanes_preds = lanes_preds.reshape(lanes_preds.shape[0],lanes_preds.shape[1],-1,self.pts_dim)


        o1_tensor = lanes_preds.unsqueeze(2).repeat(1, 1, num_query, 1,1)
        o2_tensor = lanes_preds.unsqueeze(1).repeat(1, num_query, 1, 1,1)
        topo = torch.sum(torch.abs(o1_tensor[:,:,:,-1,:]-o2_tensor[:,:,:,0,:]),dim=3)

        topo_mask = 1-torch.eye(num_query,device=topo.device)

        sigma = torch.std(topo)


        P = 2
        w = 11.5275
        lamda_1 = 0
        lamda_2 = 1

        topo = torch.exp(-torch.pow(topo,P)/(w))*topo_mask

        threshold = 0.7
        num_query = topo.shape[1]

        # Inter_type = True
        Inter_type = False
        if Inter_type:
            topo = topo.permute(0,2,1)

        # mean_pts_lst = []
        idx = preds_dicts['all_cls_scores'][-1,:,:,:].sigmoid()>0.3
        
        idx = torch.nonzero(idx.reshape(-1)).reshape(-1)
        # lanes_preds.[idx]
        select_lanes_preds = lanes_preds[:,idx]

        select_lanes_preds_new = select_lanes_preds.clone()
        num_select_lanes = select_lanes_preds.shape[1]
        select_topo = topo[:,idx][:,:,idx]


        torch.nonzero(select_topo > threshold)[:,1:]

        for i in range(num_select_lanes):
            pts = []
            pts_start = torch.nonzero(select_topo[0][i]>threshold)
            num_pts = torch.nonzero(select_topo[0][i]>threshold).shape[0]
            # if  !=None:
            if Inter_type:
                pts.append(select_lanes_preds[0][i][0].unsqueeze(0))
            else:
                pts.append(select_lanes_preds[0][i][-1].unsqueeze(0))

            for j in range(num_pts):
                if Inter_type:
                    pts.append(select_lanes_preds[0][pts_start[j][0]][-1].unsqueeze(0))
                else:
                    pts.append(select_lanes_preds[0][pts_start[j][0]][0].unsqueeze(0))

            mean_pts = torch.mean(torch.cat(pts,dim=0), dim=0)

            if Inter_type:
                select_lanes_preds_new[0,i,0,:] = mean_pts
            else:
                select_lanes_preds_new[0,i,-1,:] = mean_pts

            for j in range(num_pts):
                if Inter_type:
                    select_lanes_preds_new[0,pts_start[j][0],-1,:] = mean_pts
                else:
                    select_lanes_preds_new[0,pts_start[j][0],0,:] = mean_pts

        lanes_preds[:,idx] = select_lanes_preds_new

        if Inter_type:
            topo = topo.permute(0,2,1)



        Inter_type = True
        if Inter_type:
            topo = topo.permute(0,2,1)

        # mean_pts_lst = []
        idx = preds_dicts['all_cls_scores'][-1,:,:,:].sigmoid()>0.3
        
        idx = torch.nonzero(idx.reshape(-1)).reshape(-1)
        # lanes_preds.[idx]
        select_lanes_preds = lanes_preds[:,idx]

        select_lanes_preds_new = select_lanes_preds.clone()
        num_select_lanes = select_lanes_preds.shape[1]
        select_topo = topo[:,idx][:,:,idx]


        torch.nonzero(select_topo > threshold)[:,1:]

        for i in range(num_select_lanes):
            pts = []
            pts_start = torch.nonzero(select_topo[0][i]>threshold)
            num_pts = torch.nonzero(select_topo[0][i]>threshold).shape[0]
            # if  !=None:
            if Inter_type:
                pts.append(select_lanes_preds[0][i][0].unsqueeze(0))
            else:
                pts.append(select_lanes_preds[0][i][-1].unsqueeze(0))

            for j in range(num_pts):
                if Inter_type:
                    pts.append(select_lanes_preds[0][pts_start[j][0]][-1].unsqueeze(0))
                else:
                    pts.append(select_lanes_preds[0][pts_start[j][0]][0].unsqueeze(0))

            mean_pts = torch.mean(torch.cat(pts,dim=0), dim=0)
            # mean_pts_lst.append(mean_pts)
            if Inter_type:
                select_lanes_preds_new[0,i,0,:] = mean_pts
            else:
                select_lanes_preds_new[0,i,-1,:] = mean_pts

            for j in range(num_pts):
                if Inter_type:
                    select_lanes_preds_new[0,pts_start[j][0],-1,:] = mean_pts
                else:
                    select_lanes_preds_new[0,pts_start[j][0],0,:] = mean_pts

        lanes_preds[:,idx] = select_lanes_preds_new

        if Inter_type:
            topo = topo.permute(0,2,1)

        preds_dicts['all_lanes_preds'][-1] = lanes_preds.reshape(lanes_preds.shape[0], num_query, -1)

        lanes_preds = preds_dicts['all_lanes_preds'][-1]
        lanes_preds = lanes_preds.reshape(lanes_preds.shape[0],lanes_preds.shape[1],-1,self.pts_dim)

        o1_tensor = lanes_preds.unsqueeze(2).repeat(1, 1, num_query, 1,1)
        o2_tensor = lanes_preds.unsqueeze(1).repeat(1, num_query, 1, 1,1)
        topo = torch.sum(torch.abs(o1_tensor[:,:,:,-1,:]-o2_tensor[:,:,:,0,:]),dim=3)

        topo_mask = 1-torch.eye(num_query,device=topo.device)
        sigma = torch.std(topo)

        P = 2
        w = 11.5275
        lamda_1 = 1
        lamda_2 = 1

        topo = torch.exp(-torch.pow(topo,P)/(w))*topo_mask

        distance_topo = topo.detach().cpu().numpy()
        Sim_topo = preds_dicts['all_lclc_preds'][-1].squeeze(-1).sigmoid().detach().cpu().numpy()
        all_lclc_preds = lamda_1*Sim_topo + lamda_2*distance_topo
        all_lclc_preds = [_ for _ in all_lclc_preds]

        all_lcte_preds = preds_dicts['all_lcte_preds'][-1].squeeze(-1).sigmoid().detach().cpu().numpy()
        all_lcte_preds = [_ for _ in all_lcte_preds]

        lane_preds_dicts, pts_preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(lane_preds_dicts)
        lane_ret_list = []
        for i in range(num_samples):
            preds = lane_preds_dicts[i]
            lanes = preds['lane3d']
            scores = preds['scores']
            labels = preds['labels']
            lane_ret_list.append([lanes, scores, labels])

        pts_ret_list = []
        for i in range(num_samples):
            preds = pts_preds_dicts[i]
            lanes = preds['lane3d']
            scores = preds['scores']
            labels = preds['labels']
            pts_ret_list.append([lanes, scores, labels])

        return lane_ret_list, pts_ret_list, all_lclc_preds, all_lcte_preds
