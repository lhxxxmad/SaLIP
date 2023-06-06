import os
from collections import OrderedDict
from random import shuffle
from types import SimpleNamespace
import torch
from torch import nn
from torch._C import device
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import CrossModel, Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, Slip
from .transformer import DualTransformer
from .transformer.mutihead_attention import MultiheadAttention
from .transformer.xpool import XPool
from .loss import ivc_loss, cal_nll_loss, rec_loss
import numpy as np
allgather = AllGather.apply
allgather2 = AllGather2.apply
import pdb
import math
import itertools
class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class SLIP(nn.Module):
    def __init__(self, config):
        super(SLIP, self).__init__()

        self.config = config
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        self.slip = Slip(k=config.centerK,
                         stage_num=config.stage_num,
                         momentum=config.momentum,
                         lamd=config.lamd,
                         beta=config.beta)

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
        # self.cross_attn_ = MultiheadAttention(embed_dim, transformer_heads//2, dropout=0.1)
        self.attn = nn.MultiheadAttention(embed_dim, transformer_heads//2, dropout=0.1)
        # self.attn = MultiheadAttention(embed_dim, transformer_heads//2, dropout=0.1)
        ## embedding mode
        self.embd_mode = config.embd_mode
        self.interact_mode = config.interact_mode
        self.num_props = config.num_props
        self.xpool = XPool(config)
        self.text_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))
        self.video_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))
        self.text_saliency_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, config.max_words))
        self.video_saliency_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, config.max_frames))
        self.temporal_order_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, config.max_frames))

        self.moment_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, self.num_props * 2))

        ## ===> generate Gaussian masks
        self.mse_loss = nn.MSELoss(reduction='none')
        self.rec_loss = True
        self.do_gauss = config.do_gauss

        
        self.dropout = 0.1
        self.sal_pred = config.sal_predictor
        self.saliency_video_trans = nn.Transformer(embed_dim, transformer_heads//2, num_decoder_layers=config.sal_trans_num_layers, num_encoder_layers=config.sal_trans_num_layers, dim_feedforward= embed_dim << 1)
        self.saliency_text_trans = nn.Transformer(embed_dim, transformer_heads//2, num_decoder_layers=config.sal_trans_num_layers, num_encoder_layers=config.sal_trans_num_layers, dim_feedforward= embed_dim << 1)
        self.rec_video_trans1 = DualTransformer(num_heads=transformer_heads//2, num_decoder_layers1=config.rec_trans_num_layers1, num_decoder_layers2=config.rec_trans_num_layers1)
        self.rec_text_trans1 = DualTransformer(num_heads=transformer_heads//2, num_decoder_layers1=config.rec_trans_num_layers2, num_decoder_layers2=config.rec_trans_num_layers2)
        self.rec_video_trans2 = DualTransformer(num_heads=transformer_heads//2, num_decoder_layers1=config.rec_trans_num_layers1, num_decoder_layers2=config.rec_trans_num_layers1)
        self.rec_text_trans2 = DualTransformer(num_heads=transformer_heads//2, num_decoder_layers1=config.rec_trans_num_layers2, num_decoder_layers2=config.rec_trans_num_layers2)
        self.temporal_trans = DualTransformer(num_heads=transformer_heads//2, num_decoder_layers1=config.tmp_trans_num_layers, num_decoder_layers2=config.tmp_trans_num_layers)
        self.video_vec = nn.Parameter(torch.zeros(embed_dim).float(), requires_grad=True)
        self.text_vec = nn.Parameter(torch.zeros(embed_dim).float(), requires_grad=True)
        self.fc_gauss = nn.Linear(embed_dim, self.num_props*2)
        self.sigma = config.sigma
        self.gamma = config.gamma
        self.word_fc = nn.Linear(embed_dim, embed_dim)
        self.fc_comp = nn.Linear(embed_dim, vocab_size)
        self.mask_video_vec = nn.Parameter(torch.zeros(embed_dim).float(), requires_grad=True)
        self.mask_text_vec = nn.Parameter(torch.zeros(embed_dim).float(), requires_grad=True)
        self.max_epoch = config.epochs
        self.use_negative = True
        self.temp_loss_weight = config.temp_loss_weight
        self.rec_loss_weight = config.rec_loss_weight
        self.ret_loss_weight = config.ret_loss_weight
        self.trans = DualTransformer()
        ## ===> end of generate Gaussian masks


        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config
            
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn(config)
        
        self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)

        ## ===> Initialization trick [HARD CODE]
        new_state_dict = OrderedDict()
                
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        self.load_state_dict(new_state_dict, strict=False)  # only update new state (seqTransf/seqLSTM/tightTransf)
        ## <=== End of initialization trick

    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):

        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # B x N_v x 3 x H x W - >  (B x N_v) x 3 x H x W
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, video_feat, cls, text_mask, video_mask = self.get_text_video_feat(text_ids, text_mask, video, video_mask, shaped=True, gauss=self.do_gauss)

        if self.training:
            if torch.cuda.is_available():  # batch merge here
                idx = allgather(idx, self.config)
                text_feat = allgather(text_feat, self.config)
                video_feat = allgather(video_feat, self.config)
                text_mask = allgather(text_mask, self.config)
                video_mask = allgather(video_mask, self.config)
                cls = allgather(cls, self.config)
                torch.distributed.barrier()  # force sync

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
            
            # loss = 0.
            retrieval_loss, text_weight, video_weight, props = self.get_similarity_loss(text_feat, cls, video_feat, text_mask, video_mask, shaped=True)

            if self.do_gauss:
                # remove learnable token
                text_feat = text_feat[:, : -1]
                video_feat = video_feat[:, : -1]
                text_mask = text_mask[:, : -1]
                video_mask = video_mask[:, : -1]

            # rec_text_loss, rec_video_loss , temporal_loss = 0,0,0
            rec_video_loss, rec_text_loss = self.get_rec_loss(text_feat, video_feat, text_mask, video_mask, text_weight, video_weight)
            temporal_loss = self.get_temporal_order_loss(text_feat, video_feat, text_mask, video_mask, text_weight, video_weight)
            # moment-text rec
            rec_mt, rec_tm = self.get_moment_text_rec(text_feat, video_feat, text_mask, video_mask, props, text_weight)
            # final_loss = self.ret_loss_weight * retrieval_loss + self.rec_loss_weight * (rec_video_loss + rec_text_loss)/2.0 + self.temp_loss_weight * temporal_loss
            final_loss = self.ret_loss_weight * retrieval_loss + self.rec_loss_weight * (rec_video_loss + rec_text_loss)/2.0 + (rec_mt + rec_tm)/2.0
            final_loss_dict = {'final_loss': final_loss.item(), 
                                'retrieval_loss': self.ret_loss_weight * retrieval_loss.item(), 
                                'rec_video_loss': self.rec_loss_weight * rec_video_loss.item(), 
                                'rec_text_loss': self.rec_loss_weight * rec_text_loss.item(),
                                'rec_mt_loss': rec_mt.item(),
                                'rec_tm_loss':rec_tm.item(),
                                # 'temporal_loss': self.temp_loss_weight * temporal_loss.item()
                                }
            
            return final_loss, final_loss_dict
        else:
            return None
            
    def get_moment_text_rec(self, text_feat, video_feat, text_mask, video_mask, props, text_weight):
        bsz, frame_len, T = video_feat.shape
        props = torch.sigmoid(props).view(bsz*self.num_props, 2)
        gauss_center = props[:, 0]
        gauss_width = props[:, 1]

        video_feat = video_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, -1, T)
        video_mask = video_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)    
        
        masked_text, masked_vec_text = self._mask_feat(text_feat, text_mask.sum(1), text_weight, mask_rate=self.config.text_mask_rate)
        text_feat = text_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, -1, T)
        text_mask = text_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)      
        masked_text = masked_text.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, -1, T)
        masked_vec_text = masked_vec_text.squeeze().unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        text_weight = text_weight.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)        

        gauss_weight = self.generate_gauss_weight(frame_len, gauss_center, gauss_width)
        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]
        mask_moment, masked_vec_video = self._mask_moment(video_feat, video_mask, gauss_center, gauss_width)

        rec_text = self.rec_text_trans2(video_feat, None, masked_text, None, decoding=2, gauss_weight=pos_weight)[1]
        rec_video = self.rec_video_trans2(text_feat, None, mask_moment, None,  decoding=2, gauss_weight=text_weight)[1]

        rec_video_loss = self.mse_loss(rec_video, video_feat)
        rec_text_loss = self.mse_loss(rec_text, text_feat)

        rec_video_loss = rec_video_loss * masked_vec_video * pos_weight.unsqueeze(2)
        rec_text_loss = rec_text_loss * masked_vec_text.unsqueeze(-1) * text_weight.unsqueeze(2)
        return rec_text_loss.mean(), rec_video_loss.mean()

    def BCE_loss(self, logits, labels, mask):
        labels = labels.type_as(logits)
        loss_per_location = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        mask = mask.type_as(logits)
        loss = (loss_per_location * mask).sum() / (mask.sum() + 1e-4)
        return loss

    def get_temporal_order_loss(self, text_feat, video_feat, text_mask, video_mask, text_weight, video_weight):
        B, T, D = video_feat.shape
        shuffle_idx = torch.from_numpy(np.random.permutation(np.arange(T))).to(video_feat.device)
        shuffle_video_feat = video_feat[:,shuffle_idx]
        shuffle_video_mask = video_mask[:,shuffle_idx]
        recover_idx = torch.argsort(shuffle_idx)
        shuffle_out = self.temporal_trans(text_feat, None, shuffle_video_feat, None,  decoding=2, gauss_weight=text_weight)[1]
        # shuffle_out = self.temporal_trans(text_feat, text_mask, shuffle_video_feat, shuffle_video_mask,  decoding=2, gauss_weight=text_weight)[1]
        temporal_order_pred = self.temporal_order_fc(shuffle_out).squeeze()
        shuffle_idx = shuffle_idx.unsqueeze(0).repeat(B,1)
        # temporal_loss = self.BCE_loss(temporal_order_pred, shuffle_idx, shuffle_video_mask)
        # temporal_loss = nn.BCEWithLogitsLoss(pos_weight=shuffle_video_mask)(temporal_order_pred, shuffle_idx.float())
        # temporal_loss = nn.CrossEntropyLoss()(temporal_order_pred, shuffle_idx)
        temporal_loss = self.nll_loss(temporal_order_pred, shuffle_idx, shuffle_video_mask, video_weight).mean()
        return temporal_loss

    def nll_loss(self, logit, idx, mask, weights=None):
        eps = 0.1
        acc = (logit.max(dim=-1)[1]==idx).float()
        mean_acc = (acc * mask).sum() / mask.sum()
        # logit = logit.log_softmax(dim=-1).log_softmax(dim=-2)
        logit = logit.log_softmax(dim=-1)
        nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -logit.sum(dim=-1)
        nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss

        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        if weights is not None:
            nll_loss = nll_loss * weights

        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
        # nll_loss = (nll_loss * weights).sum(dim=-1)

        return nll_loss

    def get_rec_loss(self, text_feat, video_feat, text_mask, video_mask, text_weight, video_weight):
        # text_weight = allgather(text_weight, self.config)
        # video_weight = allgather(video_weight, self.config)

        # random mask_rete 
        # text_mask_rate = np.random.uniform(0, 1.0)
        # video_mask_rate = np.random.uniform(0, 1.0)
        # masked_video = self._mask_feat(video_feat, video_mask.sum(1), video_weight, mask_rate=video_mask_rate )
        # masked_text = self._mask_feat(text_feat, text_mask.sum(1), text_weight, mask_rate=text_mask_rate)

        # mask_rete 
        masked_video, masked_vec_video = self._mask_feat(video_feat, video_mask.sum(1), video_weight, mask_rate=self.config.video_mask_rate)
        masked_text, masked_vec_text = self._mask_feat(text_feat, text_mask.sum(1), text_weight, mask_rate=self.config.text_mask_rate)

        # #  p = random
        # masked_video = self._mask_feat(video_feat, video_mask.sum(1), mask_rate=self.config.video_mask_rate)
        # masked_text = self._mask_feat(text_feat, text_mask.sum(1), mask_rate=self.config.text_mask_rate)
        # w/ mask
        # rec_video = self.rec_trans(masked_video, video_mask, text_feat, text_mask, decoding=1, gauss_weight=text_weight)[1]
        # rec_text = self.rec_trans(video_feat, video_mask, masked_text, text_mask, decoding=2, gauss_weight=video_weight)[1]

        rec_video = self.rec_video_trans1(text_feat, None, masked_video, None,  decoding=2, gauss_weight=text_weight)[1]
        rec_text = self.rec_text_trans1(video_feat, None, masked_text, None, decoding=2, gauss_weight=video_weight)[1]

        # w/o gauss weight
        # rec_video = self.rec_video_trans(text_feat, None, masked_video, None,  decoding=2, gauss_weight=text_weight)[1]
        # rec_text = self.rec_text_trans(video_feat, None, masked_text, None, decoding=2, gauss_weight=video_weight)[1]

        rec_video_loss = self.mse_loss(rec_video, video_feat)
        rec_text_loss = self.mse_loss(rec_text, text_feat)

        rec_video_loss = rec_video_loss * masked_vec_video * video_weight.unsqueeze(2)
        rec_text_loss = rec_text_loss * masked_vec_text * text_weight.unsqueeze(2)
        return rec_video_loss.mean(), rec_text_loss.mean()

    def get_similarity_logits(self, text_feat, cls, video_feat, text_mask, video_mask, video_attention_mask=None, gauss=False):
        video_mask = video_mask.squeeze()
        text_mask = text_mask.squeeze()
        # crossmodal_cyc_loss, inmodal_cyc_loss, inmodal_contras_loss = 0., 0., 0.
        text_weight, video_weight = None, None
        cls, video_feat = cls.contiguous(), video_feat.contiguous()

        # if video_attention_mask is not None:
        #     video_attention_mask = video_attention_mask.contiguous()
        #     if self.training:
        #         video_attention_mask = allgather(video_attention_mask, self.config)
        #     video_feat = video_feat * (video_attention_mask.unsqueeze(-1) + 1e-10)
            # video_feat = video_feat / video_attention_mask.sum(dim=-1, keepdim=True)
        if self.embd_mode == 'slip':
            v_weight = torch.einsum('ad,bvd->abv', [cls, video_feat])
            v_weight = torch.softmax(v_weight / self.config.temp, dim=-1)
            if video_attention_mask is None:
                v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask])
            else:
                # v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask * video_attention_mask / video_attention_mask.sum(dim=-1, keepdim=True) ])
                v_weight = torch.einsum('abv,bv->abv', [v_weight, video_attention_mask])
                # v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask])
            video_feat = torch.einsum('abv,bvd->abd', [v_weight, video_feat])
            a, d = cls.size()
            video_feat = video_feat.contiguous().view(-1, d)
            all_embedding = torch.cat((video_feat, cls), dim=0)
            all_embedding = self.slip(all_embedding, if_train=self.training)
            video_feat = all_embedding[:video_feat.size(0), :]
            text_feat = all_embedding[video_feat.size(0):, :]
            
            _t_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            _v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
            
            _v_feat = _v_feat.view(a, -1, d)
            
            retrieve_logits = torch.einsum('ad,abd->ab', [_t_feat, _v_feat])
        elif self.embd_mode == 'xpool':
            video_feat = self.xpool(cls, video_feat, video_attention_mask=video_attention_mask)
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
            text_feat = cls / cls.norm(dim=-1, keepdim=True)            
            retrieve_logits = torch.bmm(text_feat.unsqueeze(1), video_feat.permute(1,2,0)).squeeze(1)

        elif self.embd_mode == 'cyc':
            bs = video_feat.size()[0]
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
            video_feat = self.get_video_avg_feat(video_feat, video_mask)
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

            text_feat = self.get_text_sep_feat(text_feat, text_mask).squeeze(1)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            retrieve_logits = torch.matmul(text_feat, video_feat.t())
            # if self.training:
            #     text_logits = torch.matmul(text_feat, text_feat.t()) * self.clip.logit_scale.exp()
            #     video_logits = torch.matmul(video_feat, video_feat.t()) * self.clip.logit_scale.exp()

            #     crossmodal_cyc_loss = (retrieve_logits* self.clip.logit_scale.exp() - retrieve_logits.T* self.clip.logit_scale.exp()).square().mean() / (self.clip.logit_scale.exp() * self.clip.logit_scale.exp())
            #     inmodal_cyc_loss = (video_logits - text_logits).square().mean() / (self.clip.logit_scale.exp() * self.clip.logit_scale.exp())

            #     inmodal_contras_loss = (self.loss_fct(text_logits) + self.loss_fct(video_logits) ) / 2 

        elif self.embd_mode == 'wti':
            video_mask = video_mask.squeeze()
            text_mask = text_mask.squeeze()

            ############################
            # SA
            # cross-attn
            # ################################################################
            B_t, N_t, D = text_feat.shape
            B_v, N_v, D = video_feat.shape
            if B_t > B_v:
                pad_feat = torch.zeros(1, N_v, D).to(video_feat.device)
                pad_feat = pad_feat.repeat(B_t-B_v, 1, 1)
                video_feat = torch.cat([video_feat, pad_feat], dim=0)
            if B_t < B_v:
                pad_feat = torch.zeros(1, N_t, D).to(text_feat.device)
                pad_feat = pad_feat.repeat(B_v-B_t, 1, 1)
                text_feat = torch.cat([text_feat, pad_feat], dim=0)
            try:
                if self.sal_pred == 'ca+mlp':
                    cross_text_feat = self.attn(text_feat.permute(1,0,2), video_feat.permute(1,0,2), video_feat.permute(1,0,2))[0].permute(1,0,2)
                    cross_video_feat = self.attn(video_feat.permute(1,0,2), text_feat.permute(1,0,2), text_feat.permute(1,0,2))[0].permute(1,0,2)
                elif self.sal_pred == 'trans':
                    cross_text_feat = self.saliency_text_trans(video_feat.permute(1,0,2), text_feat.permute(1,0,2)).permute(1,0,2)
                    # cross_text_feat = self.rec_text_trans1(text_feat, None, video_feat, None, decoding=1)[1]
                    cross_video_feat = self.saliency_video_trans(text_feat.permute(1,0,2), video_feat.permute(1,0,2)).permute(1,0,2)
                    # cross_video_feat = self.rec_video_trans1(video_feat, None, text_feat, None,  decoding=1)[1]
                elif self.sal_pred == 'mlp':
                    cross_text_feat = text_feat
                    cross_video_feat = video_feat
                elif self.sal_pred == 'sa+mlp':
                    cross_text_feat = self.attn(text_feat.permute(1,0,2), text_feat.permute(1,0,2), text_feat.permute(1,0,2))[0].permute(1,0,2)
                    cross_video_feat = self.attn(video_feat.permute(1,0,2), video_feat.permute(1,0,2), video_feat.permute(1,0,2))[0].permute(1,0,2)
            except:
                pdb.set_trace()
            if B_t < B_v:
                text_feat = text_feat[: B_t, ::]
                cross_text_feat = cross_text_feat[: B_t, ::]
            if B_t > B_v:
                video_feat = video_feat[: B_v, ::]
                cross_video_feat = cross_video_feat[: B_v, ::]

            # saliency token
            if gauss:
                # text_weight = self.text_weight_fc(cross_text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t
                # text_weight =  self.text_saliency_fc(cross_text_feat[:,-1])
                # video_weight =  self.video_saliency_fc(cross_video_feat[:,-1])
                props = self.moment_fc(cross_video_feat[:,-1])
                cross_video_feat = cross_video_feat[:, : -1]
                cross_text_feat = cross_text_feat[:, : -1]

                text_weight = self.text_weight_fc(cross_text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t
                video_weight = self.video_weight_fc(cross_video_feat).squeeze(2) # B_v x N_v x D -> B_v x N_v

                text_feat = text_feat[:, : -1]
                video_feat = video_feat[:, : -1]
                text_mask = text_mask[:, : -1]
                video_mask = video_mask[:, : -1]
            else:
                # MLP
                # text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t
                # Cross-Attn
                text_weight = self.text_weight_fc(cross_text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t

                # MLP
                # video_weight = self.video_weight_fc(video_feat).squeeze(2) # B_v x N_v x D -> B_v x N_v
                # Cross-Attn
                video_weight = self.video_weight_fc(cross_video_feat).squeeze(2) # B_v x N_v x D -> B_v x N_v
            
            text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
            text_weight = torch.softmax(text_weight, dim=-1)  # B_t x N_t
            # text_weight = torch.sigmoid(text_weight)  # B_t x N_t            


            video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
            video_weight = torch.softmax(video_weight, dim=-1)  # B_v x N_v
            # video_weight = torch.sigmoid(video_weight)  # B_v x N_v
            # ################################################################
            
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

            retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
            retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
            retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
            text_sum = text_mask.sum(-1)
            video_sum = video_mask.sum(-1)

            if self.interact_mode == 'FGW':
                # weighted token-wise interaction
                t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
                t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

                v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
                v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
                retrieve_logits = (t2v_logits + v2t_logits) / 2.0            
            elif self.interact_mode == 'FGM':
                t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
                v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv

                t2v_logits = torch.einsum('abt,at->abt', [t2v_logits, text_weight])
                v2t_logits = torch.einsum('abv,bv->abv', [v2t_logits, video_weight])
                
                t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(1))
                v2t_logits = torch.sum(v2t_logits, dim=2) / (video_sum.unsqueeze(0))
                retrieve_logits = (t2v_logits + v2t_logits) / 2.0

            elif self.interact_mode == 'CGW':
                text_feat = torch.einsum('atd,at->atd', [text_feat, text_mask])
                text_feat = torch.einsum('atd,at->ad', [text_feat, text_weight])
                video_feat = torch.einsum('bvd,bv->bvd', [video_feat, video_mask])
                video_feat = torch.einsum('bvd,bv->bd', [video_feat, video_weight])
                retrieve_logits = torch.einsum('ad,bd->ab', [text_feat, video_feat])

            elif self.interact_mode == 'CGM':
                text_feat = torch.einsum('atd,at->atd', [text_feat, text_mask])
                video_feat = torch.einsum('bvd,bv->bvd', [video_feat, video_mask])

                text_feat = torch.einsum('atd,at->atd', [text_feat, text_weight]) + text_feat
                video_feat = torch.einsum('bvd,bv->bvd', [video_feat, video_weight]) + video_feat

                text_feat = torch.sum(text_feat, dim=1) / (text_sum.unsqueeze(1))
                video_feat = torch.sum(video_feat, dim=1) / (video_sum.unsqueeze(1))
                retrieve_logits = torch.einsum('ad,bd->ab', [text_feat, video_feat])
                
        if self.training:
            return retrieve_logits, retrieve_logits.T, text_weight, video_weight, props
        return retrieve_logits, retrieve_logits.T

    def _mean_pooling_for_similarity_visual(self, video_feat, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_out
    def _mean_pooling_for_similarity_sequence(self, text_feat, text_mask):
        text_mask_un = text_mask.to(dtype=torch.float).unsqueeze(-1)
        text_mask_un[:, 0, :] = 0.
        text_feat = text_feat * text_mask_un
        text_out = torch.sum(text_feat, dim=1) / torch.sum(text_mask_un, dim=1, dtype=torch.float)
        return text_out

    def get_text_feat(self, text_ids, text_mask, shaped=False, gauss=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)

        if gauss:
            text_vec = self.text_vec.view(1, 1, -1).expand(bs_pair, 1, -1).float() #
            # text_ids = torch.cat([text_ids, text_vec], dim=1) #
            mask = torch.zeros(bs_pair).byte().cuda().view(-1,1)
            text_mask = torch.cat([text_mask, mask], dim=1) #
            cls, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask, text_vec=text_vec)
        else:
            cls, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        # cls, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)

        cls, text_feat = cls.float(), text_feat.float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))
        cls = cls.view(bs_pair, -1, cls.size(-1)).squeeze(1)
        return text_feat, cls, text_mask

    def get_video_feat(self, video, video_mask, shaped=False, gauss=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)
        
        bs_pair, n_v = video_mask.size()
        video_feat = self.clip.encode_image(video, return_hidden=True)[0].float()
        video_feat = video_feat.float().view(bs_pair, -1, video_feat.size(-1))

        # add Gaussian masks
        if gauss:
            video_vec = self.video_vec.view(1, 1, -1).expand(bs_pair, 1, -1) #
            video_feat = torch.cat([video_feat, video_vec], dim=1) #
            mask = torch.zeros(bs_pair).byte().cuda().view(-1,1)
            video_mask = torch.cat([video_mask, mask], dim=1) #
        video_feat = self.agg_video_feat(video_feat, video_mask, self.agg_module)
        return video_feat, video_mask

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, shaped=False, gauss=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)
        
        text_feat, cls, text_mask = self.get_text_feat(text_ids, text_mask, shaped=True, gauss=gauss)
        video_feat, video_mask = self.get_video_feat(video, video_mask, shaped=True, gauss=gauss)
        # get proposal
        # _, trans_out = self.trans(video_feat, video_mask, text_feat, text_mask, decoding=1)

        return text_feat, video_feat, cls, text_mask, video_mask

    def generate_gauss_weight(self, props_len, center, width):

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]

    def _mask_feat(self, feat, feat_len, weights=None, mask_rate = 0.3):
        
        masked_vec = []
        for i, l in enumerate(feat_len):
            l = int(l)
            # num_masked_vec = max(l // 3, 1) 
            num_masked_vec = max(int(l * mask_rate), 1) 
            masked_vec.append(torch.zeros([feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().detach().numpy() if weights is not None else None
            # choices = np.random.choice(np.arange(l), num_masked_vec, replace=False)
            choices = np.random.choice(np.arange(l), num_masked_vec, replace=False, p=p)
            # choices = torch.topk(weights[i, :l], k=num_masked_vec)[1]
            masked_vec[-1][choices] = 1

        masked_vec = torch.stack(masked_vec, 0).unsqueeze(-1)
        # out_feat = feat.masked_fill(masked_vec == 1, float("-inf"))
        out_feat = feat.masked_fill(masked_vec == 1, 0)
        return out_feat, masked_vec
    def _mask_moment(self, video_feat, video_mask, center, width):
        video_len = video_mask.sum(1)

        star, end =  torch.clamp(center-width/2, min=0), torch.clamp(center+width/2, max=1)
        star, end = (video_len * star).to(torch.int), (video_len * end).to(torch.int)
        masked_vec = torch.zeros(video_mask.shape).byte().cuda()

        for i, l in enumerate(video_len):
            if star[i] < end[i]:
                masked_vec[i][star[i]:end[i]] = 1
            elif star[i] == end[i]:
                masked_vec[i][star[i]] = 1
            else:
                masked_vec[i][end[i]:star[i]] = 1
        masked_vec = masked_vec.unsqueeze(-1)
        video_feat = video_feat.masked_fill(masked_vec == 1, 0)
        return video_feat, masked_vec

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    def get_text_sep_feat(self, text_feat, text_mask):
        text_feat = text_feat.contiguous()
        text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
        text_feat = text_feat.unsqueeze(1).contiguous()
        return text_feat

    def agg_video_feat(self, video_feat, video_mask, agg_module):
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training: self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
            video_feat = video_feat + video_feat_original
        return video_feat


    def get_marginal_loss(self, cosx, m, s):
        sinx = torch.sqrt(1.0 - torch.pow(cosx, 2))
        cosm = math.cos(m)
        sinm = math.sin(m)
        return (cosx * cosm - sinx * sinm)/s
        
    def get_similarity_loss(self, text_feat, cls, video_feat, text_mask, video_mask, shaped=False, video_attention_mask=None):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        t2v_logits, v2t_logits, text_weight, video_weight, props = self.get_similarity_logits(text_feat, cls, video_feat, text_mask, video_mask, video_attention_mask=video_attention_mask, gauss=self.do_gauss)
        
        logit_scale = self.clip.logit_scale.exp()
        # pdb.set_trace()
        t2v_logits = self.get_marginal_loss(t2v_logits, 0.25, 0.05)/logit_scale
        v2t_logits = self.get_marginal_loss(v2t_logits, 0.25, 0.05)/logit_scale

        loss_t2v = self.loss_fct(t2v_logits * logit_scale)
        loss_v2t = self.loss_fct(v2t_logits * logit_scale)
        

        loss = (loss_t2v + loss_v2t) / 2

        return loss, text_weight, video_weight, props

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()