# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

import cv2
from tqdm import tqdm

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--output', help='Where to save the output images', type=str, metavar='DIR')

@click.option('--cmp', type = bool, metavar = 'BOOL', is_flag = True)
@click.option('--itp', type = bool, metavar = 'BOOL', is_flag = True)
@click.option('--src', type = bool, metavar = 'BOOL', is_flag = True)
@click.option('--itp-qual', type = bool, metavar = 'BOOL', is_flag = True)
@click.option('--dual-itp', type = bool, metavar = 'BOOL', is_flag = True)
@click.option('--num', help='Total number of generated images. Only when --seeds is unspecified', type=int)

@click.option('--network2', help = 'The 2nd network providing delta w for ensemble', type = str)
@click.option('--ensemble-mode', help = 'Ensemble mode for multiple delta w [default: mean]', type = click.Choice(['mean', 'sum', 'mean_norm']))


def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    output: str,
    class_idx: Optional[int],
    projected_w: Optional[str],

    cmp: bool,
    itp: bool,
    itp_qual: bool,
    dual_itp: bool,
    src: bool,
    num: int,

    network2: str,
    ensemble_mode: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    if num is None:
        num = 20    

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        pkl = legacy.load_network_pkl(f)
        G = pkl['G_ema'].to(device) # type: ignore
        if cmp or itp:
            training_set = dnnlib.util.construct_class_by_name(**pkl['training_set_kwargs'])
            training_img = training_set.load_all_images().to(device).to(torch.float32) / 127.5 - 1
            if training_img.shape[0] < num:
                training_img = torch.cat([training_img, -torch.ones(num - training_img.shape[0], *training_img.shape[1:]).to(device)], dim = 0)
            elif training_img.shape[0] > num:
                training_img = training_img[:num]

    if network2 is not None:
        print('Loading 2nd network from "%s" with mode %s' % (network2, ensemble_mode))
        with dnnlib.util.open_url(network2) as f2:
            pkl2 = legacy.load_network_pkl(f2)
            G2 = pkl2['G_ema'].to(device)
            if ensemble_mode is not None:
                # ensemble_mode = 'mean'
                cos_sim = {}
                # print(G.synthesis.const_delta_w4)
                for n, p in G.named_parameters():
                    if 'const_delta_w' in n:
                        res = int(n.split('const_delta_w')[-1])
                        cur_dw = getattr(G.synthesis, f'const_delta_w{res}')
                        cur2_dw = getattr(G2.synthesis, f'const_delta_w{res}')
                        cos_sim[res] = (cur_dw.abs().mean().item(), cur2_dw.abs().mean().item(), torch.nn.functional.cosine_similarity(cur_dw, cur2_dw, dim = -1).mean().item())
                        if ensemble_mode == 'mean':
                            cur_dw += cur2_dw
                            cur_dw /= 2
                        elif ensemble_mode == 'sum':
                            cur_dw += cur2_dw
                        elif ensemble_mode == 'mean_norm':
                            target_norm = (cur_dw.norm(dim = -1, keepdim = True) + cur2_dw.norm(dim = -1, keepdim = True)) / 2
                            cur_dw += cur2_dw
                            cur_dw /= cur_dw.norm(dim = -1, keepdim = True)
                            assert torch.all((cur_dw.norm(dim = -1) - 1.0).abs() < 1e-6)
                            cur_dw *= target_norm

                    # print(G.synthesis.const_delta_w4)
                print('Ensembled resolution & cos sim')
                print('res:\t1_l1\t2_l1\tcos')
                for res, cos in cos_sim.items():
                    print('%d\t%.4f\t%.4f\t%.4f' % (res, *cos))
                        

    if cmp and itp:
        print('Error: cmp OR itp')
        exit(1)

    if output is not None and (not output.endswith('.png') and not output.endswith('.jpg')):
        os.makedirs(output, exist_ok=True)
    if (cmp or itp or dual_itp) and output is None:
        assert network_pkl[-4:] == '.pkl'
        kimg = network_pkl[-10:-4]
        if cmp:
            mode = 'cmp'
        elif itp:
            mode = 'itp'
        elif dual_itp:
            mode = 'dual_itp'
        output = os.path.join(os.path.dirname(network_pkl), f'{mode}{kimg}.jpg')

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{output}/proj{idx:02d}.png')
        return

    if seeds is None:
        seeds = [x for x in range(num)]
        #ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    if cmp:
        assert hasattr(G, 'transfer') and G.transfer in ['const_delta_w', 'adapt_delta_w', 'learn_alpha_cdw', 'const_delta_w_alpha']
        img_list = []
        src_img_list = []
        coef_list = []
        for seed_idx, seed in tqdm(enumerate(seeds), desc = 'Generating'):
            # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            # src img
            ws = G.mapping(z, label, truncation_psi = truncation_psi)
            src_img = G.synthesis(ws, mod_intensity = 0.0, noise_mode = noise_mode)

            # tgt img
            tgt_img = G.synthesis(ws, noise_mode = noise_mode)
            if G.transfer in ['learn_alpha_cdw', 'const_delta_w_alpha']:
                tgt_no_alpha_img = G.synthesis(ws, noise_mode = noise_mode, mod_alpha = False)
                black = np.zeros((tgt_img.shape[2] * 2, tgt_img.shape[3], 3), dtype = np.uint8)
                for i in range(len(G.synthesis.block_resolutions)):
                    res = G.synthesis.block_resolutions[i]
                    alpha_net = getattr(G.synthesis, f'alpha_net{res}' if G.transfer in ['learn_alpha_cdw', 'const_delta_w_alpha'] else 'alpha_net')
                    alpha = alpha_net(ws[:, 0, :])
                    text = 'res%4d:%6.2f' % (res, alpha[0, 0].item()) if seed_idx == 0 else '%6.2f' % alpha[0, 0].item()
                    black = cv2.putText(black, text, (20 if seed_idx == 0 else 70, 50 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                black = torch.from_numpy(black).to(tgt_img.device).permute(2, 0, 1) / 127.5 - 1
                tgt_img = torch.cat([tgt_no_alpha_img, tgt_img, black.unsqueeze(0)], dim = 2)

            src_img_list.append(src_img[0])
            img_list.append(tgt_img[0])

        src_imgs = torch.cat(src_img_list, dim = 2)
        tgt_imgs = torch.cat(img_list, dim = 2)
            
        canvas = [torch.cat([x for x in training_img], dim = 2), src_imgs, tgt_imgs]
        canvas = torch.cat(canvas, dim = 1)
        canvas = (canvas.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        if not output.endswith('.jpg'):
            output += '.jpg'
        PIL.Image.fromarray(canvas.cpu().numpy(), 'RGB').save(f'{output}')

    elif itp:
        assert hasattr(G, 'transfer') and G.transfer in ['const_delta_w', 'adapt_delta_w', 'learn_alpha_cdw', 'const_delta_w_alpha']
        # itp_lambda_list = [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        itp_lambda_list = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
        # itp_lambda_list = [0.0, 1.0, 1.25, 1.5, 1.75, 2.0]
        img_list = [[] for _ in range(len(itp_lambda_list))]
        for seed_idx, seed in tqdm(enumerate(seeds), desc = 'Generating'):
            # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            # src img
            ws = G.mapping(z, label, truncation_psi = truncation_psi)
            
            for i in range(len(itp_lambda_list)):
                img = G.synthesis(ws, mod_intensity = itp_lambda_list[i], noise_mode = noise_mode)
                img_list[i].append(img[0])
           
        for i in range(len(itp_lambda_list)):
            img_list[i] = torch.cat(img_list[i], dim = 2)
        img_list = torch.cat(img_list, dim = 1)
        img_list = torch.cat([torch.cat([x for x in training_img], dim = 2), img_list], dim = 1)

        black = np.zeros((img_list.shape[1], 128, 3), dtype = np.uint8)
        resolution = img.shape[-1]
        for i in range(len(itp_lambda_list)):
            black = cv2.putText(black, '%.2f' % itp_lambda_list[i], (30, resolution * (i + 1) + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        black = torch.from_numpy(black).to(img_list.device).permute(2, 0, 1) / 127.5 - 1

        canvas = torch.cat([black, img_list], dim = 2)
        canvas = (canvas.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        if not output.endswith('.jpg'):
            output += '.jpg'
        PIL.Image.fromarray(canvas.cpu().numpy(), 'RGB').save(f'{output}')

    elif dual_itp:
        assert hasattr(G, 'transfer') and G.transfer in ['const_delta_w', 'adapt_delta_w', 'learn_alpha_cdw', 'const_delta_w_alpha']
        print('Loading 2nd network from "%s" with mode %s' % (network2, ensemble_mode))
        with dnnlib.util.open_url(network2) as f2:
            pkl2 = legacy.load_network_pkl(f2)
            G2 = pkl2['G_ema'].to(device)

        itp_lambda_list = [0.0, 0.25, 0.5, 0.75, 1.0]
        res_list = [4, 8, 16, 32, 64, 128, 256]
        with torch.no_grad():
            for seed_idx, seed in tqdm(enumerate(seeds), desc = 'Generating'):
                img_list = [[] for _ in range(len(itp_lambda_list))]
                # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
                # src img
                ws = G.mapping(z, label, truncation_psi = truncation_psi)
                
                for i in range(len(itp_lambda_list)):
                    for j in itp_lambda_list:
                        cur = 0
                        dw1_backup = {}
                        for res in res_list:
                            dw1 = getattr(G.synthesis, f'const_delta_w{res}')
                            dw2 = getattr(G2.synthesis, f'const_delta_w{res}')
                            dw1_backup[res] = dw1.clone()
                            # dw1 = dw1 * itp_lambda_list[i] + dw2 * j
                            dw1 *= itp_lambda_list[i]
                            dw1 += dw2 * j

                        img = G.synthesis(ws, mod_intensity = 1.0, noise_mode = noise_mode)
                        img_list[i].append(img[0])

                        for res in res_list:
                            setattr(G.synthesis, f'const_delta_w{res}', torch.nn.Parameter(dw1_backup[res]))
            
                for i in range(len(itp_lambda_list)):
                    img_list[i] = torch.cat(img_list[i], dim = 2)
                img_list = torch.cat(img_list, dim = 1)
                # img_list = torch.cat([torch.cat([x for x in training_img], dim = 2), img_list], dim = 1)

                canvas = (img_list.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(canvas.cpu().numpy(), 'RGB').save(f'{output}/seed{seed:04d}.jpg')

    elif itp_qual:
        assert hasattr(G, 'transfer') and G.transfer in ['const_delta_w', 'adapt_delta_w', 'learn_alpha_cdw', 'const_delta_w_alpha']
        itp_lambda_list = [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        for seed_idx, seed in tqdm(enumerate(seeds), desc = 'Generating'):
            canvas = []
            # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            # src img
            ws = G.mapping(z, label, truncation_psi = truncation_psi)
            
            for i in range(len(itp_lambda_list)):
                img = G.synthesis(ws, mod_intensity = itp_lambda_list[i], noise_mode = noise_mode)
                canvas.append(img)
           
            canvas = torch.cat(canvas, dim = 3)
            img = (canvas.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{output}/seed{seed:04d}.jpg')

    else:
        for seed_idx, seed in tqdm(enumerate(seeds), desc = 'Generating'):
            # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{output}/seed{seed:04d}.png')
            if src:
                src_img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, mod_intensity = 0.0)
                src_img = (src_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(src_img[0].cpu().numpy(), 'RGB').save(f'{output}/seed{seed:04d}_src.png')
            


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
