import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER

from projects.bevformer.modules.decoder import CustomMSDeformableAttention
from projects.bevformer.modules.spatial_cross_attention import \
    MSDeformableAttention3D
from projects.bevformer.modules.temporal_self_attention import \
    TemporalSelfAttention


@TRANSFORMER.register_module()
class TopoPointTransformerDecoderOnly(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 decoder=None,
                 embed_dims=256,
                 pts_dim=3,
                 **kwargs):
        super(TopoPointTransformerDecoderOnly, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.fp16_enabled = False
        self.pts_dim = pts_dim
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, self.pts_dim)
        self.pts_reference_points = nn.Linear(self.embed_dims, self.pts_dim)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_embed,
                object_query_embed,
                pts_object_query_embeds,
                bev_h,
                bev_w,
                ptlc_branches=None,
                lclc_branches=None,
                lcte_branches=None,
                reg_branches=None,
                pt_reg_branches = None,
                te_feats=None,
                te_cls_scores=None,
                **kwargs):

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        pts_query_pos, pts_query = torch.split(
            pts_object_query_embeds, self.embed_dims, dim=1)

        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        pts_query_pos = pts_query_pos.unsqueeze(0).expand(bs, -1, -1)
        pts_query = pts_query.unsqueeze(0).expand(bs, -1, -1)

        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()

        pts_reference_points = self.pts_reference_points(pts_query_pos)
        pts_reference_points = pts_reference_points.sigmoid()

        init_reference_out = reference_points
        init_pts_reference_out = pts_reference_points

        query = query.permute(1, 0, 2)
        pts_query = pts_query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        pts_query_pos = pts_query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        
        inter_states, pts_inter_states, inter_references, inter_ptlc_rel, inter_lclc_rel, inter_lcte_rel, lc_outputs_coords, pts_outputs_coords,lc_outputs_score,pts_outputs_score = self.decoder(
            query=query,
            pts_query=pts_query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            pts_query_pos=pts_query_pos,
            reference_points=reference_points,
            pts_reference_points=pts_reference_points,
            ptlc_branches=ptlc_branches,
            lclc_branches=lclc_branches,
            lcte_branches=lcte_branches,
            reg_branches = reg_branches,
            pt_reg_branches = pt_reg_branches,
            pts_num = self.pts_dim,
            te_feats=te_feats,
            te_cls_scores=te_cls_scores,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return inter_states, pts_inter_states, init_reference_out, inter_references_out, inter_ptlc_rel, inter_lclc_rel, inter_lcte_rel, lc_outputs_coords, pts_outputs_coords,lc_outputs_score,pts_outputs_score
