from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cgitb import text

import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from modules.until_module import PreTrainedModel, AllGather, CrossEn, Emcl
from modules.module_cross import CrossConfig, Transformer as TransformerClip
from modules.module_clip import CLIP, convert_weights
from modules.loss import CrossEn

from .PDE import DisTrans
import torch.nn.functional as F
logger = logging.getLogger(__name__)
allgather = AllGather.apply


class EMCL4QAPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, cross_config, *inputs, **kwargs):
        super(EMCL4QAPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-B/16")
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None,
                                                 task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        contain_frame_position = False
        for key in state_dict.keys():
            if key.find("frame_position_embeddings") > -1:
                contain_frame_position = True
                break
        if contain_frame_position is False:
            for key, val in clip_state_dict.items():
                if key == "positional_embedding":
                    state_dict["frame_position_embeddings.weight"] = val.clone()
                    continue
                if key.find("transformer.resblocks") == 0:
                    num_layer = int(key.split(".")[2])
                    # cut from beginning
                    if num_layer < task_config.cross_num_hidden_layers:
                        state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                        continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class EMCL4QA(EMCL4QAPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(EMCL4QA, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1
        self.dropout = nn.Dropout(0.1)

        self.emcl = Emcl(k=task_config.K,
                         stage_num=task_config.stage_num,
                         momentum=task_config.momentum,
                         lamd=task_config.lamd,
                         beta=task_config.beta)

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b
                            in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers - cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        cross_config.max_position_embeddings = context_length
        self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
        self.transformerClip = TransformerClip(width=transformer_width,
                                                   layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )

        hidden_size = transformer_width * 8
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size,
                      hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, task_config.num_labels)
        )

        self.v_proj = nn.Linear(transformer_width, 4 * transformer_width)
        self.t_proj = nn.Linear(transformer_width, 4 * transformer_width)
        self.loss_fct = CrossEn()

        if vit:
            self.mean_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.mean_proj = nn.Linear(embed_dim, embed_dim)

        self.v_w1 = nn.Sequential(
            nn.Linear(transformer_width, transformer_width),
            nn.ReLU(True),
            nn.Linear(embed_dim, 1)
        )

        self.t_w1 = nn.Sequential(
            nn.Linear(transformer_width, transformer_width),
            nn.ReLU(True),
            nn.Linear(embed_dim, 1)
        )
        self.v_w2 = nn.Sequential(
            nn.Linear(transformer_width, transformer_width),
            nn.ReLU(True),
            nn.Linear(embed_dim, 1)
        )

        self.t_w2 = nn.Sequential(
            nn.Linear(transformer_width, transformer_width),
            nn.ReLU(True),
            nn.Linear(embed_dim, 1)
        )
        self.attn1 = nn.MultiheadAttention(embed_dim, transformer_heads//2, dropout=0.1)
        self.attn2 = nn.MultiheadAttention(embed_dim, transformer_heads//2, dropout=0.1)
        self.sample_num = 2
        # self.dropout = 0.1

        self.dist_video_trans = DisTrans(transformer_width, transformer_heads)
        self.dist_text_trans = DisTrans(transformer_width, transformer_heads)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,
                labels=None):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        # , txt_mu, txt_logsigma, vid_mu, vid_logsigma
        text_feat, video_feat = self.get_sequence_video_feat(input_ids,
                                                                                                         token_type_ids,
                                                                                                         attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)


        # video_feat = self.v_proj(video_feat)
        # text_feat = self.t_proj(text_feat)

        # input = torch.cat((video_feat, text_feat), dim=1)
        # pooled_output = self.dropout(input)
        # logits = self.classifier(pooled_output)

        if self.training:
            labels = allgather(labels, self.task_config)
            
            video_feat = allgather(video_feat, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            text_feat = allgather(text_feat, self.task_config)
            attention_mask = allgather(attention_mask, self.task_config)
            torch.distributed.barrier()
            logits1 = self.get_cl_output1(text_feat, video_feat, attention_mask, video_mask)
            logits1, loss1 = self.calc_loss(logits1, labels)
            # b0, n = video_feat.size()
            # b1, n = text_feat.size()
            # all_embedding = torch.cat((video_feat, text_feat), dim=0)
            # all_embedding = self.emcl(all_embedding, if_train=self.training)
            # video_feat = all_embedding[:b0, :].view(b0, n)
            # text_feat = all_embedding[b0:, :].view(b1, n)
            # video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
            # text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            # logit_scale = self.clip.logit_scale.exp()
            # retrieve_logits = logit_scale * torch.matmul(text_feat, video_feat.t())
            # sim_loss = (self.loss_fct(retrieve_logits) + self.loss_fct(retrieve_logits.T)) / 2
            # loss = loss + sim_loss * 0.5

            t2v_logits, v2t_logits= self.get_dist_logits(text_feat, video_feat, attention_mask, video_mask)
            logit_scale = self.clip.logit_scale.exp()

            # t2v_logits = self.get_marginal_loss(t2v_logits, 0.25, 0.05)/logit_scale
            # v2t_logits = self.get_marginal_loss(v2t_logits, 0.25, 0.05)/logit_scale

            loss_t2v = self.loss_fct(t2v_logits * logit_scale)
            loss_v2t = self.loss_fct(v2t_logits * logit_scale)
            
            sim_loss = (loss_t2v + loss_v2t) / 2

            # logits2 = self.get_cl_output2(text_feat, video_feat, attention_mask, video_mask)
            # logits2, loss2 = self.calc_loss(logits2, labels)

            loss = loss1+ sim_loss
            
            return loss
        else:
            logits1 = self.get_cl_output1(text_feat, video_feat, attention_mask, video_mask)

            logits2 = self.get_cl_output2(text_feat, video_feat, attention_mask, video_mask)
            return logits1 + logits2

    def calc_loss(self, logits, labels):
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                        logits.view(-1, self.task_config.num_labels),
                        labels.view(-1))
        else:
            loss = 0
        return logits, loss

    def get_text_feat(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        text_feat = self.clip.encode_text(input_ids, return_hidden=True)[0].float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))
        return text_feat

    def get_video_feat(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        video_feat = self.clip.encode_image(video, video_frame=video_frame, return_hidden=True)[0].float()
        video_feat = video_feat.view(bs_pair, -1, video_feat.size(-1))
        
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
        video_feat = self.mean_proj(video_feat)
        video_feat = video_feat + video_feat_original

        return video_feat

    def get_sequence_video_feat(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        text_feat = self.get_text_feat(input_ids, token_type_ids, attention_mask, shaped=True)
        video_feat = self.get_video_feat(video, video_mask, shaped=True, video_frame=video_frame)

        text_feat, video_feat = text_feat.contiguous(), video_feat.contiguous()
        # if self.training:
        #     video_feat = allgather(video_feat, self.task_config)
        #     video_mask = allgather(video_mask, self.task_config)
        #     text_feat = allgather(text_feat, self.task_config)
        #     attention_mask = allgather(attention_mask, self.task_config)
        #     torch.distributed.barrier()

        # text_weight = torch.softmax(self.t_w(text_feat).squeeze(2), dim=-1)  # BxN_t
        # video_weight = torch.softmax(self.v_w(video_feat).squeeze(2), dim=-1)  # BxN_t
    
        # # probability distribution sampling
        # B,N,C = text_feat.shape
        # # txt_mu, txt_logsigma, _ = self.dist_text_trans(text_feat, weight=text_weight)
        # txt_mu, txt_logsigma, _ = self.dist_text_trans(text_feat, weight=None)
        # samples = [txt_mu]
        # for _ in range(self.sample_num-1):
        #     eps = torch.randn(B, N, C, device=txt_mu.device)
        #     sample = txt_mu + torch.exp(txt_logsigma) * eps
        #     samples.append(sample)
        # # pdb.set_trace()
        # dis_text_feat = torch.cat(samples).view(B, self.sample_num, N, C).mean(dim=1)
        # text_feat = text_feat + F.dropout(dis_text_feat, p=0.1)
        # # text_feat = self.dis_fc1(text_feat)
        # # text_mask = text_mask.unsqueeze(1).expand(B, self.sample_num, -1).reshape(B * self.sample_num, N)

        # B,N,C = video_feat.shape
        # # vid_mu, vid_logsigma, _ = self.dist_video_trans(video_feat, weight=video_weight)
        # vid_mu, vid_logsigma, _ = self.dist_video_trans(video_feat, weight=None)
        # samples = [vid_mu]
        # for _ in range(self.sample_num-1):
        #     eps = torch.randn(B, N, C, device=vid_mu.device)
        #     sample = vid_mu + torch.exp(vid_logsigma) * eps
        #     samples.append(sample)
        # dis_video_feat = torch.cat(samples).view(B, self.sample_num, N, C).mean(dim=1)
        # video_feat = video_feat + F.dropout(dis_video_feat, p=0.1)
        # video_feat = self.dis_fc2(video_feat)
        # video_mask = video_mask.unsqueeze(1).expand(B, self.sample_num, -1).reshape(B * self.sample_num, N)


        # video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        # video_feat = video_feat * video_mask_un
        # attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        # text_feat = text_feat * attention_mask_un


        # text_feat = torch.einsum(" atc,at->ac ", [text_feat, text_weight])
        # video_feat = torch.einsum(" atc,at->ac ", [video_feat, video_weight])
        return text_feat, video_feat #, txt_mu, txt_logsigma, vid_mu, vid_logsigma

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

    def get_cl_output1(self, sequence_hidden, visual_hidden, attention_mask, video_mask):

        text_weight = torch.softmax(self.t_w1(sequence_hidden).squeeze(2), dim=-1)  # BxN_t
        video_weight = torch.softmax(self.v_w1(visual_hidden).squeeze(2), dim=-1)  # BxN_t

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_hidden * video_mask_un

        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        sequence_output = sequence_hidden * attention_mask_un

        sequence_output = torch.einsum(" atc,at->ac ", [sequence_output, text_weight])
        visual_output = torch.einsum(" atc,at->ac ", [visual_output, video_weight])

        visual_output = self.v_proj(visual_output)
        sequence_output = self.t_proj(sequence_output)
        input = torch.cat((visual_output, sequence_output), dim=1)
        pooled_output = self.dropout(input)
        logits = self.classifier(pooled_output)

        return logits

    def get_cl_output2(self, sequence_hidden, visual_hidden, attention_mask, video_mask):

        text_weight = torch.softmax(self.t_w2(sequence_hidden).squeeze(2), dim=-1)  # BxN_t
        video_weight = torch.softmax(self.v_w2(visual_hidden).squeeze(2), dim=-1)  # BxN_t

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_hidden * video_mask_un

        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        sequence_output = sequence_hidden * attention_mask_un

        sequence_output = torch.einsum(" atc,at->ac ", [sequence_output, text_weight])
        visual_output = torch.einsum(" atc,at->ac ", [visual_output, video_weight])

        visual_output = self.v_proj(visual_output)
        sequence_output = self.t_proj(sequence_output)
        input = torch.cat((visual_output, sequence_output), dim=1)
        pooled_output = self.dropout(input)
        logits = self.classifier(pooled_output)

        return logits

    def get_dist_logits(self, text_feat, video_feat, attention_mask, video_mask):

        cross_text_feat = self.attn2(text_feat.permute(1,0,2), video_feat.permute(1,0,2), video_feat.permute(1,0,2))[0].permute(1,0,2)
        cross_video_feat = self.attn2(video_feat.permute(1,0,2), text_feat.permute(1,0,2), text_feat.permute(1,0,2))[0].permute(1,0,2)

        text_weight = self.t_w2(cross_text_feat).squeeze(2)  # B_t x N_t x D -> B_t x N_t
        video_weight = self.v_w2(cross_video_feat).squeeze(2) # B_v x N_v x D -> B_v x N_v

        text_weight.masked_fill_(torch.tensor((1 - attention_mask), dtype=torch.bool), float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)  # B_t x N_t
        # text_weight = torch.sigmoid(text_weight)  # B_t x N_t            

        video_weight = video_weight.masked_fill(torch.tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
        video_weight = torch.softmax(video_weight, dim=-1)  # B_v x N_v


        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, attention_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        text_sum = attention_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        # t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        # v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        # t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(1))
        # v2t_logits = torch.sum(v2t_logits, dim=2) / (video_sum.unsqueeze(0))

        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])

        retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        return retrieve_logits, retrieve_logits.T

def Wasserstein2(mu1, sigma1, mu2, sigma2): # 2W距离，传入图片和文本的均值和标准差
    bs1 = mu1.shape[0]
    bs2 = mu2.shape[0]
    mu1 = torch.stack([mu1]*bs2, dim=1)
    sigma1 = torch.stack([sigma1]*bs2, dim=1)
    mu2 = torch.stack([mu2]*bs1, dim=0)
    sigma2 = torch.stack([sigma2]*bs1, dim=0)
    p1 = torch.sum(torch.pow(mu1 - mu2, 2), dim=-1)
    p2 = torch.sum(torch.pow(sigma1 - sigma2, 2), dim=-1)
    return p1+p2, p1

def compute_dis_contrast(txt_mu, txt_sigma, vid_mu, vid_sigma, negative_scale = 1/2000, shift = 4, temp = 0.01):
    # loss_fct = CrossEn()
    # vid_mu = vid_mu[:, 0]
    # vid_sigma = torch.exp(vid_logsigma[:, 0])
    # txt_mu = txt_mu[:, 0]
    # txt_sigma = torch.exp(txt_logsigma[:, 0])

    # pl_module.log('con/img_sigma_mean', torch.mean(vid_sigma), on_step=True)
    # pl_module.log('con/txt_sigma_mean', torch.mean(txt_sigma), on_step=True)
    
    bs = vid_mu.shape[0]
    # phase = "train" if pl_module.training else "val"

    # gather
    # allgather = AllGather_multi.apply
    # vid_mu = allgather(vid_mu)
    # txt_mu = allgather(txt_mu)
    # vid_sigma = allgather(vid_sigma)
    # txt_sigma = allgather(txt_sigma)

    W2_distance, mu_distance = Wasserstein2(vid_mu, vid_sigma, txt_mu, txt_sigma)
    similarity = (-negative_scale * W2_distance + shift) / temp

    labels = torch.arange(bs).to(similarity.device)
    loss = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(0, 1), labels)) / 2
    # loss = loss_fct(similarity) + loss_fct(similarity.transpose(0, 1)
    
    # pl_module.log(f"contrast/{phase}/loss", loss)
    # pl_module.log("temperature", pl_module.temp)

    # ret = {'contrast_loss': loss}

    return loss