import copy
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import Linear, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout 
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, FEEDFORWARD_NETWORK,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.topopoint.models.dense_heads.relationship_head import MLP

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TopoPointSGNNDecoder(TransformerLayerSequence):

    def __init__(self, pc_range,*args, return_intermediate=False, **kwargs):
        super(TopoPointSGNNDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.w = nn.Parameter(torch.tensor([10],dtype=torch.float32))
        # self.lamda_1 = nn.Parameter(torch.tensor([1],dtype=torch.float32))
        # self.lamda_2 = nn.Parameter(torch.tensor([1],dtype=torch.float32))
        self.P = nn.Parameter(torch.tensor([2],dtype=torch.float32))
        self.pt_w = nn.Parameter(torch.tensor([10],dtype=torch.float32))

        self.pt_lamda_1 = nn.Parameter(torch.tensor([1],dtype=torch.float32))
        self.pt_lamda_2 = nn.Parameter(torch.tensor([1],dtype=torch.float32))
        self.pt_P = nn.Parameter(torch.tensor([2],dtype=torch.float32))
        # self.mlp = MLP(self.embed_dims*2, self.embed_dims, self.embed_dims, 3)

            
    def forward(self,
                query,
                pts_query,
                *args,
                reference_points=None,
                pts_reference_points=None,
                pts_cls_branches=None,
                cls_branches=None,
                ptlc_branches=None,
                mlp_branches=None,
                lclc_branches=None,
                lcte_branches=None,
                reg_branches = None,
                pt_reg_branches = None,
                pts_num = 3,
                key_padding_mask=None,
                te_feats=None,
                te_cls_scores=None,
                **kwargs):

        output = query
        pts_output = pts_query
        pts_intermediate_score = []
        lc_intermediate_score = []
        intermediate = []
        pts_intermediate = []
        intermediate_reference_points = []
        intermediate_pts_reference_points = []
        intermediate_ptlc_rel = []
        intermediate_lclc_rel = []
        intermediate_lcte_rel = []
        lc_outputs_coords = []
        pts_outputs_coords = []
        num_pts_query = pts_query.size(0)
        num_query = query.size(0)
        num_te_query = te_feats.size(2)
        prev_ptlc_adj = torch.zeros((query.size(1), num_pts_query, num_query),
                                  dtype=query.dtype, device=query.device)
        prev_lclc_adj = torch.zeros((query.size(1), num_query, num_query),
                                  dtype=query.dtype, device=query.device)
        prev_lcte_adj = torch.zeros((query.size(1), num_query, num_te_query),
                                  dtype=query.dtype, device=query.device)

        
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            pts_reference_points_input = pts_reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            
            output, pts_output = layer(
                output,
                pts_output,
                *args,
                reference_points=reference_points_input,
                pts_reference_points=pts_reference_points_input,
                key_padding_mask=key_padding_mask,
                te_query=te_feats[lid],
                te_cls_scores=te_cls_scores[lid],
                ptlc_adj=prev_ptlc_adj,
                lclc_adj=prev_lclc_adj,
                lcte_adj=prev_lcte_adj,
                **kwargs)


            output = output.permute(1, 0, 2)
            pts_output = pts_output.permute(1, 0, 2)
            bs, ins_query, _ = pts_output.shape

            pt_reference = inverse_sigmoid(pts_reference_points)
            pt_tmp = pt_reg_branches[lid](pts_output)

            bs, num_pts, _ = pt_tmp.shape
            pt_tmp = pt_tmp.view(bs, num_pts, -1, 3)
            pt_tmp = pt_tmp + pt_reference.unsqueeze(2)
            pt_tmp = pt_tmp.sigmoid()

            pts_reference_points = pt_tmp.squeeze(2).detach().clone()

            pt_coord = pt_tmp.clone()
            pt_coord[..., 0] = pt_coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            pt_coord[..., 1] = pt_coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            pt_coord[..., 2] = pt_coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2] 
            pts_outputs_coord = pt_coord.view(bs, num_pts, -1).contiguous()

            lc_reference = inverse_sigmoid(reference_points)
            lc_tmp = reg_branches[lid](output)

            bs, num_lc, _ = lc_tmp.shape
            lc_tmp = lc_tmp.view(bs, num_lc, -1, 3)
            lc_tmp = lc_tmp + lc_reference.unsqueeze(2)
            lc_tmp = lc_tmp.sigmoid()
            coord = lc_tmp.clone()
            coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2] 
            lc_outputs_coord = coord.view(bs, num_lc, -1).contiguous()

            pt_tensor = pt_coord.detach().unsqueeze(2).repeat(1, 1, num_query, 1,1)
            lc_tensor = coord.detach().unsqueeze(1).repeat(1, num_pts_query, 1, 1,1)
            distance_1 = torch.sum(torch.abs(pt_tensor[:,:,:,0,:] - lc_tensor[:,:,:,0,:]), dim=3)
            distance_2 = torch.sum(torch.abs(pt_tensor[:,:,:,0,:] - lc_tensor[:,:,:,-1,:]), dim=3)
            ptlc_distance = torch.min(distance_1,distance_2)
            ptlc_topo = torch.exp(-torch.pow(ptlc_distance,self.pt_P)/(self.pt_w))

            o1_tensor = coord.detach().unsqueeze(2).repeat(1, 1, num_query, 1,1)
            o2_tensor = coord.detach().unsqueeze(1).repeat(1, num_query, 1, 1,1)
            topo = torch.sum(torch.abs(o1_tensor[:,:,:,-1,:]-o2_tensor[:,:,:,0,:]),dim=3)
            topo_mask = 1-torch.eye(num_query,device=topo.device)
            sigma = torch.std(topo)

            topo = torch.exp(-torch.pow(topo,self.P)/(self.w))*topo_mask

            ptlc_rel_out = ptlc_branches[lid](pts_output, output)
            lclc_rel_out = lclc_branches[lid](output, output)
            lclc_rel_adj = lclc_rel_out.squeeze(-1).sigmoid()
            ptlc_rel_adj = ptlc_rel_out.squeeze(-1).sigmoid()

            prev_ptlc_adj = self.pt_lamda_1*ptlc_rel_adj.detach() + self.pt_lamda_2*ptlc_topo
            # prev_ptlc_adj = ptlc_topo

            pts_score = pts_cls_branches[lid](pts_output)
            lc_score = cls_branches[lid](output)

            prev_lclc_adj = topo

            lcte_rel_out = lcte_branches[lid](output, te_feats[lid])
            lcte_rel_adj = lcte_rel_out.squeeze(-1).sigmoid()
            prev_lcte_adj = lcte_rel_adj.detach()
            
            output = output.permute(1, 0, 2)
            pts_output = pts_output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)
                pts_intermediate.append(pts_output)
                intermediate_reference_points.append(reference_points)
                intermediate_ptlc_rel.append(ptlc_rel_out)
                intermediate_lclc_rel.append(lclc_rel_out)
                intermediate_lcte_rel.append(lcte_rel_out)
                lc_outputs_coords.append(lc_outputs_coord)
                pts_outputs_coords.append(pts_outputs_coord)
                lc_intermediate_score.append(lc_score)
                pts_intermediate_score.append(pts_score)
                

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(pts_intermediate), torch.stack(
                intermediate_reference_points), torch.stack(
                intermediate_ptlc_rel), torch.stack(
                intermediate_lclc_rel), torch.stack(
                intermediate_lcte_rel), torch.stack(
                lc_outputs_coords), torch.stack(
                pts_outputs_coords),torch.stack(
                lc_intermediate_score),torch.stack(pts_intermediate_score)

        return output, pts_output, reference_points, ptlc_rel_out, lclc_rel_out, lcte_rel_out, lc_outputs_coords, pts_outputs_coords,lc_intermediate_score,pts_intermediate_score


@TRANSFORMER_LAYER.register_module()
class SGNNDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(SGNNDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs)
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(
        #     ['self_attn', 'norm', 'cross_attn', 'ffn'])

    
    
    def forward(self,
                query,
                pts_query,
                key=None,
                value=None,
                query_pos=None,
                pts_query_pos=None,
                key_pos=None,
                attn_masks=None,
                reference_points=None,
                pts_reference_points=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                te_query=None,
                te_cls_scores=None,
                
                ptlc_adj=None,
                lclc_adj=None,
                lcte_adj=None,
                **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        pts_identity = pts_query
        num_lane_query = query.shape[0]
        num_pts_query = pts_query.shape[0]
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:

            if layer == 'self_attn' and attn_index == 0:

                ptlc_query = torch.cat((pts_query, query),dim=0)
                temp_key = temp_value = ptlc_query
                ptlc_query_pos = torch.cat((pts_query_pos, query_pos), dim=0)
                ptlc_query = self.attentions[attn_index](
                    ptlc_query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=ptlc_query_pos,
                    key_pos=ptlc_query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1

                query = ptlc_query[num_pts_query:,:,:]
                pts_query = ptlc_query[:num_pts_query,:,:]
                pts_identity = pts_query
                identity = query


            elif layer == 'norm' and norm_index == 0 and attn_index == 1:

                query = self.norms[norm_index](query)
                norm_index += 1
                pts_query = self.norms[norm_index](pts_query)
                norm_index += 1

            elif layer == 'norm' and norm_index == 2 and attn_index == 3:
                query = self.norms[norm_index](query)
                norm_index += 1
                pts_query = self.norms[norm_index](pts_query)
                norm_index += 1

            elif layer == 'norm' and norm_index == 4 and ffn_index == 1:
                query = self.norms[norm_index](query)
                norm_index += 1
                pts_query = self.norms[norm_index](pts_query)
                norm_index += 1

            elif layer == 'cross_attn' and attn_index == 1:
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    reference_points = reference_points,
                    **kwargs)
                attn_index += 1
                identity = query

                pts_query = self.attentions[attn_index](
                    pts_query,
                    key,
                    value,
                    pts_identity if self.pre_norm else None,
                    query_pos=pts_query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    reference_points = pts_reference_points,
                    **kwargs)

                attn_index += 1
                pts_identity = pts_query

            elif layer == 'ffn' and ffn_index == 0:

                pts_query, query = self.ffns[ffn_index](
                    pts_query, query, te_query, ptlc_adj, lclc_adj, lcte_adj, te_cls_scores, identity=identity if self.pre_norm else None)
                ffn_index += 1

        return query, pts_query


@FEEDFORWARD_NETWORK.register_module()
class FFN_SGNN(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.1,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 edge_weight=0.5, 
                 num_te_classes=13,
                 **kwargs):
        super(FFN_SGNN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        lane_layers = []
        pt_layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            lane_layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            pt_layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        lane_layers.append(
            Sequential(
                Linear(feedforward_channels, embed_dims), self.activate,
                nn.Dropout(ffn_drop)))
        pt_layers.append(
            Sequential(
                Linear(feedforward_channels, embed_dims), self.activate,
                nn.Dropout(ffn_drop)))
        self.lane_layers = Sequential(*lane_layers)
        self.pt_layers = Sequential(*pt_layers)
        self.edge_weight = edge_weight

        self.ptlc_gnn_layer = ptlcSkgGCNLayer(embed_dims, embed_dims, edge_weight=edge_weight)
        self.ptlc_gnn_layer_1 = ptlcSkgGCNLayer(embed_dims, embed_dims, edge_weight=edge_weight)
        
        self.lclc_gnn_layer = LclcSkgGCNLayer(embed_dims, embed_dims, edge_weight=edge_weight)
        self.lcte_gnn_layer = LcteSkgGCNLayer(embed_dims, embed_dims, num_te_classes=num_te_classes, edge_weight=edge_weight)
        self.downsample = nn.Linear(embed_dims * 2, embed_dims)

        self.gnn_dropout1 = nn.Dropout(ffn_drop)
        self.gnn_dropout2 = nn.Dropout(ffn_drop)

        self.lc_dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.pt_dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, pt_query, lc_query, te_query, ptlc_adj, lclc_adj, lcte_adj, te_cls_scores, identity=None):
        pt_out = self.pt_layers(pt_query)
        pt_out = pt_out.permute(1, 0, 2)
        lc_out = self.lane_layers(lc_query)
        lc_out = lc_out.permute(1, 0, 2)
        num_pts = pt_out.shape[1]

        ptlc_out = torch.cat((pt_out, lc_out), dim = 1)
        ptlc_features = self.ptlc_gnn_layer_1(ptlc_out, ptlc_adj)
        pt_features = ptlc_features[:,:num_pts,:]
        lc_features = ptlc_features[:,num_pts:,:]
        pt_out = pt_features
        lc_out = lc_features


        lclc_features = self.lclc_gnn_layer(lc_out, lclc_adj)
        lcte_features = self.lcte_gnn_layer(te_query, lcte_adj, te_cls_scores)

        lc_out = torch.cat([lclc_features, lcte_features], dim=-1)
        lc_out = self.activate(lc_out)
        lc_out = self.gnn_dropout1(lc_out)
        lc_out = self.downsample(lc_out)
        lc_out = self.gnn_dropout2(lc_out)
        # lc_out = lclc_features

        ptlc_out = torch.cat((pt_out, lc_out), dim = 1)
        ptlc_features = self.ptlc_gnn_layer(ptlc_out, ptlc_adj)

        pt_features = ptlc_features[:,:num_pts,:]
        lc_features = ptlc_features[:,num_pts:,:]
        pt_out = pt_features
        lc_out = lc_features

        lc_out = lc_out.permute(1, 0, 2)
        pt_out = pt_out.permute(1, 0, 2)
        
        if not self.add_identity:
            return self.pt_dropout_layer(pt_out), self.lc_dropout_layer(lc_out)
        if identity is None:
            lc_identity = lc_query
            pt_identity = pt_query
        return pt_identity + self.pt_dropout_layer(pt_out), lc_identity + self.lc_dropout_layer(lc_out)

class ptlcSkgGCNLayer(nn.Module):

    def __init__(self, in_features, out_features, edge_weight=0.5):
        super(ptlcSkgGCNLayer, self).__init__()
        self.edge_weight = edge_weight

        if self.edge_weight != 0:
            self.weight_forward = torch.Tensor(in_features, out_features)
            self.weight_forward = nn.Parameter(nn.init.xavier_uniform_(self.weight_forward))
            self.weight_backward = torch.Tensor(in_features, out_features)
            self.weight_backward = nn.Parameter(nn.init.xavier_uniform_(self.weight_backward))

        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight

    def forward(self, input, ptlc_adj):

        size = ptlc_adj.shape[1]
        pt_input = input[:,:size,:]
        lc_input = input[:,size:,:]

        support_loop = torch.matmul(input, self.weight)
        output = support_loop
        if self.edge_weight != 0:
            support_forward = torch.matmul(lc_input, self.weight_forward)
            output_forward = torch.matmul(ptlc_adj, support_forward)
            output[:,:size,:] += self.edge_weight * output_forward

            support_backward = torch.matmul(pt_input, self.weight_backward)
            output_backward = torch.matmul(ptlc_adj.permute(0, 2, 1), support_backward)
            output[:,size:,:] += self.edge_weight * output_backward

        return output


class LclcSkgGCNLayer(nn.Module):

    def __init__(self, in_features, out_features, edge_weight=0.5):
        super(LclcSkgGCNLayer, self).__init__()
        self.edge_weight = edge_weight

        if self.edge_weight != 0:
            self.weight_forward = torch.Tensor(in_features, out_features)
            self.weight_forward = nn.Parameter(nn.init.xavier_uniform_(self.weight_forward))
            self.weight_backward = torch.Tensor(in_features, out_features)
            self.weight_backward = nn.Parameter(nn.init.xavier_uniform_(self.weight_backward))

        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight

    def forward(self, input, adj):

        support_loop = torch.matmul(input, self.weight)
        output = support_loop

        if self.edge_weight != 0:
            support_forward = torch.matmul(input, self.weight_forward)

            output_forward = torch.matmul(adj, support_forward)
            output += self.edge_weight * output_forward

            support_backward = torch.matmul(input, self.weight_backward)

            output_backward = torch.matmul(adj.permute(0, 2, 1), support_backward)
            output += self.edge_weight * output_backward

        return output


class LcteSkgGCNLayer(nn.Module):

    def __init__(self, in_features, out_features, num_te_classes=13, edge_weight=0.5):
        super(LcteSkgGCNLayer, self).__init__()
        self.weight = torch.Tensor(num_te_classes, in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight

    def forward(self, te_query, lcte_adj, te_cls_scores):
        # te_cls_scores: (bs, num_te_query, num_te_classes)
        cls_scores = te_cls_scores.detach().sigmoid().unsqueeze(3)
        # te_query: (bs, num_te_query, embed_dims)
        # (bs, num_te_query, 1, embed_dims) * (bs, num_te_query, num_te_classes, 1)
        te_feats = te_query.unsqueeze(2) * cls_scores
        # (bs, num_te_classes, num_te_query, embed_dims)
        te_feats = te_feats.permute(0, 2, 1, 3)
    
        support = torch.matmul(te_feats, self.weight).sum(1)
        adj = lcte_adj * self.edge_weight
        output = torch.matmul(adj, support)
        return output
