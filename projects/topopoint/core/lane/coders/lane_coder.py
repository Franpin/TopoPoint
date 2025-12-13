import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from ..util import denormalize_3dlane
import numpy as np


@BBOX_CODERS.register_module()
class LanePseudoCoder(BaseBBoxCoder):

    def __init__(self, denormalize=False):
        self.denormalize = denormalize

    def encode(self):
        pass

    def decode_single(self, cls_scores, lane_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            lane_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """

        cls_scores = cls_scores.sigmoid()
        scores, labels = cls_scores.max(-1)
        if self.denormalize:
            final_lane_preds = denormalize_3dlane(lane_preds, self.pc_range)
        else:
            final_lane_preds = lane_preds

        predictions_dict = {
            'lane3d': final_lane_preds.detach().cpu().numpy(),
            'scores': scores.detach().cpu().numpy(),
            'labels': labels.detach().cpu().numpy()
        }

        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        pt_threshold = 0.3
        lane_threshold = 0.3

        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_lanes_preds = preds_dicts['all_lanes_preds'][-1]

        all_pts_cls_scores = preds_dicts['all_pts_cls_scores'][-1]
        all_pts_preds = preds_dicts['all_pts_preds'][-1]

        # bs, num_lanes = all_lanes_preds.shape[0], all_lanes_preds.shape[1]

        ##################################
        all_lanes_preds_new = all_lanes_preds[:,:,3:-3]

        select_lane_idx = all_cls_scores > lane_threshold
        select_lane_idx = torch.nonzero(select_lane_idx.reshape(-1)).reshape(-1)
        select_lanes_preds = all_lanes_preds_new[:,select_lane_idx,:]
        select_all_lanes_preds = all_lanes_preds[:,select_lane_idx,:]
        

        select_pt_idx = all_pts_cls_scores > pt_threshold
        select_pt_idx = torch.nonzero(select_pt_idx.reshape(-1)).reshape(-1)
        select_pts_preds = all_pts_preds[:,select_pt_idx,:]

         

        bs = select_lanes_preds.shape[0]
        
        num_lc = select_lanes_preds.shape[1]
        num_pts = select_pts_preds.shape[1]


        # if num_lc != 0 and num_pts != 0:

        #     pre_pt = select_all_lanes_preds[:,:,:3]
        #     back_pt = select_all_lanes_preds[:,:,-3:]
        #     pts_coord_repeat = select_pts_preds.unsqueeze(2).repeat(1,1,num_lc,1)
        #     pre_pt_coord_repeat = pre_pt.unsqueeze(1).repeat(1,num_pts,1,1)
        #     back_pt_coord_repeat = back_pt.unsqueeze(1).repeat(1,num_pts,1,1)

        #     pre_pt_distance = torch.sum(torch.abs(pts_coord_repeat - pre_pt_coord_repeat),3)
        #     back_pt_distance = torch.sum(torch.abs(pts_coord_repeat - back_pt_coord_repeat),3)

        #     pre_pt_distance[:,1,:] 
        #     pre_value, pre_idx = torch.min(pre_pt_distance, dim=1)
        #     back_value, back_idx = torch.min(back_pt_distance, dim=1)

        #     select_pre_pts = select_pts_preds.reshape(bs*num_pts,-1)[pre_idx.reshape(-1),:]

        #     select_pre_value = pre_value < 2

        #     select_pre_value_idx = torch.nonzero(select_pre_value.reshape(-1)).squeeze(-1)
        #     if select_pre_value_idx.shape[0] != 0:
        #         pre_pt[:,select_pre_value_idx,:] = select_pre_pts[select_pre_value_idx,:].reshape(bs, select_pre_value_idx.shape[0], -1)


        #     select_back_pts = select_pts_preds.reshape(bs*num_pts,-1)[back_idx.reshape(-1),:]
        #     select_back_value = back_value < 2
        #     select_back_value_idx = torch.nonzero(select_back_value.reshape(-1)).squeeze(-1)

        #     if select_back_value_idx.shape[0] != 0: 
        #         back_pt[:,select_back_value_idx,:] = select_back_pts[select_back_value_idx,:].reshape(bs, select_back_value_idx.shape[0], -1)

        #     # update_pre_pt = update_pre_pt.reshape(bs, num_lc, -1)
        #     # update_back_pt = update_back_pt.reshape(bs, num_lc, -1)

        #     update_select_lanes_preds = torch.cat([pre_pt, select_lanes_preds, back_pt], dim=2)

        #     all_lanes_preds[:,select_lane_idx,:] = update_select_lanes_preds


        ########################################
            
            # from IPython import embed; embed()

            # pt_outputs_coord_1 = select_pts_preds.reshape(bs, num_pts, -1, 3)
            # lc_outputs_coord_1 = select_lanes_preds.reshape(bs, num_lc, -1,3)

            # pt_outputs_coord_repeat = pt_outputs_coord_1.unsqueeze(2).repeat(1,1,num_lc,1,1)
            # lc_outputs_coord_repeat = lc_outputs_coord_1.unsqueeze(1).repeat(1,num_pts,1,1,1)

            # L1 = lc_outputs_coord_repeat[:,:,:,-1,:] - lc_outputs_coord_repeat[:,:,:,-2,:]
            # L2 = pt_outputs_coord_repeat[:,:,:,0,:] - lc_outputs_coord_repeat[:,:,:,-1,:]

            # L3 = lc_outputs_coord_repeat[:,:,:,1,:] - lc_outputs_coord_repeat[:,:,:,0,:]
            # L4 = lc_outputs_coord_repeat[:,:,:,0,:] - pt_outputs_coord_repeat[:,:,:,0,:]

            # delta_L_end = L2 - L1
            # dist_end = torch.matmul(delta_L_end.unsqueeze(3), delta_L_end.unsqueeze(-1)).view(bs, num_pts, num_lc)
            # _ ,idx_end = torch.min(dist_end, dim=1)
            # batch_pt_outputs_coord = select_pts_preds.reshape(bs*num_pts, -1,3)
            # select_end_coord = batch_pt_outputs_coord[idx_end.reshape(-1)].reshape(bs,num_lc, -1,3)

            # delta_L_start = L4 - L3
            # dist_start = torch.matmul(delta_L_start.unsqueeze(3), delta_L_start.unsqueeze(-1)).view(bs, num_pts, num_lc)
            # _, idx_start = torch.min(dist_start, dim=1)
            # batch_pt_outputs_coord = select_pts_preds.reshape(bs*num_pts, -1,3)

            # select_start_coord = batch_pt_outputs_coord[idx_start.reshape(-1)].reshape(bs,num_lc, -1,3)

            # select_end_coord = select_end_coord.reshape(bs, num_lc, -1)
            # select_start_coord= select_start_coord.reshape(bs, num_lc, -1)
            
            # # from IPython import embed; embed()

            # select_lanes_preds = torch.cat((select_start_coord,select_lanes_preds,select_end_coord),dim=2)

            # all_lanes_preds[:,select_lane_idx,:] = select_lanes_preds


        # bs = all_lanes_preds.shape[0]
        
        # num_lc = all_lanes_preds.shape[1]
        # num_pts = all_pts_preds.shape[1]

        # pt_outputs_coord_1 = all_pts_preds.reshape(bs, num_pts, -1, 3)
        # lc_outputs_coord_1 = all_lanes_preds.reshape(bs, num_lc, -1,3)

        # pt_outputs_coord_repeat = pt_outputs_coord_1.unsqueeze(2).repeat(1,1,num_lc,1,1)
        # lc_outputs_coord_repeat = lc_outputs_coord_1.unsqueeze(1).repeat(1,num_pts,1,1,1)

        # L1 = lc_outputs_coord_repeat[:,:,:,-1,:] - lc_outputs_coord_repeat[:,:,:,-2,:]
        # L2 = pt_outputs_coord_repeat[:,:,:,0,:] - lc_outputs_coord_repeat[:,:,:,-1,:]

        # L3 = lc_outputs_coord_repeat[:,:,:,1,:] - lc_outputs_coord_repeat[:,:,:,0,:]
        # L4 = lc_outputs_coord_repeat[:,:,:,0,:] - pt_outputs_coord_repeat[:,:,:,0,:]

        # delta_L_end = L2 - L1
        # dist_end = torch.matmul(delta_L_end.unsqueeze(3), delta_L_end.unsqueeze(-1)).view(bs, num_pts, num_lc)
        # _ ,idx_end = torch.min(dist_end, dim=1)
        # batch_pt_outputs_coord = all_pts_preds.reshape(bs*num_pts, -1,3)
        # select_end_coord = batch_pt_outputs_coord[idx_end.reshape(-1)].reshape(bs,num_lc, -1,3)

        # delta_L_start = L4 - L3
        # dist_start = torch.matmul(delta_L_start.unsqueeze(3), delta_L_start.unsqueeze(-1)).view(bs, num_pts, num_lc)
        # _, idx_start = torch.min(dist_start, dim=1)
        # batch_pt_outputs_coord = all_pts_preds.reshape(bs*num_pts, -1,3)

        # select_start_coord = batch_pt_outputs_coord[idx_start.reshape(-1)].reshape(bs,num_lc, -1,3)

        # select_end_coord = select_end_coord.reshape(bs, num_lc, -1)
        # select_start_coord= select_start_coord.reshape(bs, num_lc, -1)
        
        # all_lanes_preds = torch.cat((select_start_coord,all_lanes_preds,select_end_coord),dim=2)


        batch_size = all_cls_scores.size()[0]
        lane_predictions_list = []
        pts_predictions_list = []
        for i in range(batch_size):
            lane_predictions_list.append(self.decode_single(all_cls_scores[i], all_lanes_preds[i]))
            pts_predictions_list.append(self.decode_single(all_pts_cls_scores[i], all_pts_preds[i]))
        return lane_predictions_list, pts_predictions_list
