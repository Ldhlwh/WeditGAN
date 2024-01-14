# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
        transfer = None, cl = None, lambda_cl = 0.0, tau = 0.07, reg = None, lambda_reg = 0.0):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        self.transfer = transfer
        self.cl = cl
        self.reg = reg
        self.lambda_cl = lambda_cl
        self.lambda_reg = lambda_reg
        self.tau = tau

        if self.reg in ['perp']:
            self.delta_ws = {}
            if self.reg in ['weighted_l2']:
                self.alphas = {}
            for n, p in G_synthesis.named_parameters():
                if 'const_delta_w' in n:
                    res = n.split('const_delta_w')[-1]
                    self.delta_ws[int(res)] = p
                elif 'alpha' in n:
                    res = n.split('alpha')[-1]
                    self.alphas[int(res)] = p

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        other_rtn = {'ws': ws}

        with misc.ddp_sync(self.G_synthesis, sync):
            if self.cl in ['g_feat', 'gd_feat']:
                img, feat_dict = self.G_synthesis(ws, output_feat = True)
                other_rtn['feat_dict'] = feat_dict
            elif self.reg == 'alpha_l2':
                img, alpha_dict = self.G_synthesis(ws, output_alpha = True)
                other_rtn['alpha_dict'] = alpha_dict
            else:
                img = self.G_synthesis(ws)  

        if self.cl in ['g_feat', 'gd_feat']:
            with misc.ddp_sync(self.G_synthesis, sync):
                src_img, src_feat_dict = self.G_synthesis(ws, mod_intensity = 0.0, output_feat = True)
                other_rtn['src_feat_dict'] = src_feat_dict
                other_rtn['src_img'] = src_img

        return img, other_rtn


    def run_Gs(self, z, c, sync):
        with misc.ddp_sync(self.Gs_mapping, sync):
            ws = self.Gs_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.Gs_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        other_rtn = {'ws': ws}

        with misc.ddp_sync(self.Gs_synthesis, sync):
            img = self.Gs_synthesis(ws)  

        return img, other_rtn


    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        other_rtn = {}
        with misc.ddp_sync(self.D, sync):
            if self.cl in ['gd_feat']:
                logits, feat_dict = self.D(img, c, output_feat = True)
                other_rtn['feat_dict'] = feat_dict
            else:
                logits = self.D(img, c)
        return logits, other_rtn

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, other_rtn = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits, _ = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)

                if self.reg is not None:
                    if self.reg == 'perp':
                        # expand ori_ws (some of the ori_ws are reused by one conv and one torgb)
                        ori_ws = other_rtn['ws']
                        diff_ori_ws = ori_ws - torch.cat([ori_ws[1:], ori_ws[:1]]) 
                        w_idx = 0
                        loss_reg = 0.0
                        for res in self.G_synthesis.block_resolutions:
                            block = getattr(self.G_synthesis, f'b{res}')
                            cur_diff_ori_ws = diff_ori_ws.narrow(1, w_idx, block.num_conv + block.num_torgb)
                            cur_delta_ws = self.delta_ws[res]
                            cur_loss = ((cur_diff_ori_ws * cur_delta_ws).sum(dim = -1) ** 2).sum(dim = 1)
                            # cur_loss = (torch.cosine_similarity(cur_diff_ori_ws, cur_delta_ws) ** 2).sum(dim = 1)
                            training_stats.report(f'Loss/G/perp_reg{res}', cur_loss)
                            loss_reg = loss_reg + cur_loss
                            w_idx += block.num_conv
                        loss_Gmain = loss_Gmain + self.lambda_reg * loss_reg

                    elif self.reg == 'alpha_l2':
                        loss_reg = 0.0
                        alpha_dict = other_rtn['alpha_dict']
                        for res in self.G_synthesis.block_resolutions:
                            cur_loss = (alpha_dict[res] ** 2).sum(dim = (1, 2))
                            training_stats.report(f'Loss/G/alpha_l2_reg{res}', cur_loss)
                            loss_reg = loss_reg + cur_loss
                        loss_Gmain = loss_Gmain + self.lambda_reg * loss_reg
                        
                    else:
                        raise NotImplementedError(f'{self.reg} not implemented')

                if self.cl in ['g_feat', 'gd_feat']:
                    src_feat_dict, tgt_feat_dict = other_rtn['src_feat_dict'], other_rtn['feat_dict']
                    for k in tgt_feat_dict.keys():
                        res = int(k[1:])
                        src_feat = src_feat_dict[k].flatten(1).float()
                        tgt_feat = tgt_feat_dict[k].flatten(1).float()
                        pos_exp_score = torch.exp(torch.nn.functional.cosine_similarity(src_feat, tgt_feat, dim = 1) / self.tau).unsqueeze(-1)
                        neg_exp_score = torch.zeros(pos_exp_score.shape).to(pos_exp_score.device)
                        permute_src_feat = src_feat.clone()
                        for _ in range(src_feat.shape[0] - 1):
                            permute_src_feat = torch.cat([permute_src_feat[1:], permute_src_feat[:1]], dim = 0)
                            neg_exp_score = neg_exp_score + torch.exp(torch.nn.functional.cosine_similarity(permute_src_feat, tgt_feat, dim = 1) / self.tau).unsqueeze(-1)
                        loss_cl = - torch.log(pos_exp_score / (pos_exp_score + neg_exp_score))
                        training_stats.report(f'Loss/G/cl{res}', loss_cl)
                        loss_Gmain = loss_Gmain + self.lambda_cl * loss_cl               

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward() 

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, other_rtn = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                gen_ws = other_rtn['ws']

                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, other_rtn = self.run_G(gen_z, gen_c, sync=False)
                gen_logits, tgt_d_other_rtn = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                if self.cl in ['gd_feat']:
                    _, other_rtn = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))
                    src_img = other_rtn['src_img']
                    
                    _, src_d_other_rtn = self.run_D(src_img, gen_c, sync = False)
                    src_d_feat_dict, tgt_d_feat_dict = src_d_other_rtn['feat_dict'], tgt_d_other_rtn['feat_dict']
                    for k in tgt_d_feat_dict.keys():
                        res = int(k[1:])
                        src_d_feat = src_d_feat_dict[k].flatten(1).float()
                        tgt_d_feat = tgt_d_feat_dict[k].flatten(1).float()
                        pos_exp_score = torch.exp(torch.nn.functional.cosine_similarity(src_d_feat, tgt_d_feat, dim = 1) / self.tau).unsqueeze(-1)
                        neg_exp_score = torch.zeros(pos_exp_score.shape).to(pos_exp_score.device)
                        permute_src_d_feat = src_d_feat.clone()
                        for _ in range(src_d_feat.shape[0] - 1):
                            permute_src_d_feat = torch.cat([permute_src_d_feat[1:], permute_src_d_feat[:1]], dim = 0)
                            neg_exp_score = neg_exp_score + torch.exp(torch.nn.functional.cosine_similarity(permute_src_d_feat, tgt_d_feat, dim = 1) / self.tau).unsqueeze(-1)
                        loss_cl_d = - torch.log(pos_exp_score / (pos_exp_score + neg_exp_score))
                        training_stats.report(f'Loss/D/cl_d{res}', loss_cl_d)
                        loss_Dgen = loss_Dgen + self.lambda_cl * loss_cl_d

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits, _ = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
