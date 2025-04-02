# ================================================================ #
#   __  __                  __        ___     _   ___    __        #
#  / / / /__  ___ _______ _/ /__     / _ |   | | / (_)__/ /__ ___  #
# / /_/ / _ \(_-</ __/ _ `/ / -_) - / __ / - / |/ / / _  / -_) _ \ #
# \____/ .__/___/\__/\_,_/_/\__/   /_/ |_|   |___/_/\_,_/\__/\___/ #
#     /_/                                                          #                                              
# ================================================================ #

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter('ignore', FutureWarning)
import logging
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
import transformers
transformers.logging.set_verbosity_error()

import os
import cv2
import argparse
import sys
o_path = os.getcwd()
sys.path.append(o_path)

import torch
import torch.cuda
import time
import math
import json
import imageio
import textwrap
import pyfiglet
import numpy as np
import torchvision
from PIL import Image
from einops import rearrange
from torchvision.utils import flow_to_image, save_image
from torch.nn import functional as F

from models_video.RAFT.raft_bi import RAFT_bi
from models_video.propagation_module import Propagation
from models_video.autoencoder_kl_cond_video import AutoencoderKLVideo
from models_video.unet_video import UNetVideoModel
from models_video.pipeline_upscale_a_video import VideoUpscalePipeline
from models_video.scheduling_ddim import DDIMScheduler
from models_video.color_correction import wavelet_reconstruction, adaptive_instance_normalization

from llava.llava_agent import LLavaAgent
from utils import get_video_paths, read_frame_from_videos, str_to_list
from utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from configs.CKPT_PTH import LLAVA_MODEL_PATH


if __name__ == '__main__':

    if torch.cuda.device_count() >= 2:
        UAV_device = 'cuda:0'
        LLaVA_device = 'cuda:1'
    elif torch.cuda.device_count() == 1:
        UAV_device = 'cuda:0'
        LLaVA_device = 'cuda:0'
    else:
        raise ValueError('Currently support CUDA only.')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs', 
            help='Input folder.')
    parser.add_argument('-o', '--output_path', type=str, default='./results', 
            help='Output folder.')
    parser.add_argument('-n', '--noise_level', type=int, default=120, 
            help='Noise level [0, 200] applied to the input video. A higher noise level typically results in better \
                video quality but lower fidelity. Default value: 120')
    parser.add_argument('-g', '--guidance_scale', type=int, default=6, 
            help='Classifier-free guidance scale for prompts. A higher guidance scale encourages the model to generate \
                more details. Default: 6')
    parser.add_argument('-s', '--inference_steps', type=int, default=30, # 45 will add more details
            help='The number of denoising steps. More steps usually lead to a higher quality video. Default: 30')
    parser.add_argument('-p','--propagation_steps', type=str_to_list, default=[],
            help='Propagation steps after performing denoising.')
    parser.add_argument("--a_prompt", type=str, default='best quality, extremely detailed')
    parser.add_argument("--n_prompt", type=str, default='blur, worst quality')
    parser.add_argument('--use_video_vae', action='store_true', default=False)
    parser.add_argument("--color_fix", type=str, default='None', choices=["None", "AdaIn", "Wavelet"])
    parser.add_argument('--no_llava', action='store_true', default=False)
    parser.add_argument("--load_8bit_llava", action='store_true', default=False)
    parser.add_argument('--perform_tile', action='store_true', default=False)
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--save_image', action='store_true', default=False)
    parser.add_argument('--save_suffix', type=str, default='')
    parser.add_argument('--scale_factor', type=int, default=4, 
            help='The factor by which to upscale the video (e.g., 2, 3, 4). Default: 4')
    args = parser.parse_args()

    use_llava = not args.no_llava

    print(pyfiglet.figlet_format("Upscale-A-Video", font="slant"))

    ## ---------------------- load models ----------------------
    ## load upsacale-a-video
    print('Loading Upscale-A-Video')

    # load low_res_scheduler, text_encoder, tokenizer
    pipeline = VideoUpscalePipeline.from_pretrained("./pretrained_models/upscale_a_video", torch_dtype=torch.float16)

    # load vae
    if args.use_video_vae:
        pipeline.vae = AutoencoderKLVideo.from_config("./pretrained_models/upscale_a_video/vae/vae_video_config.json")
        pretrained_model = "./pretrained_models/upscale_a_video/vae/vae_video.bin"
        pipeline.vae.load_state_dict(torch.load(pretrained_model, map_location="cpu"))
    else:
        pipeline.vae = AutoencoderKLVideo.from_config("./pretrained_models/upscale_a_video/vae/vae_3d_config.json")
        pretrained_model = "./pretrained_models/upscale_a_video/vae/vae_3d.bin"
        pipeline.vae.load_state_dict(torch.load(pretrained_model, map_location="cpu"))

    # load unet
    pipeline.unet = UNetVideoModel.from_config("./pretrained_models/upscale_a_video/unet/unet_video_config.json")
    pretrained_model = "./pretrained_models/upscale_a_video/unet/unet_video.bin"
    pipeline.unet.load_state_dict(torch.load(pretrained_model, map_location="cpu"), strict=True)
    pipeline.unet = pipeline.unet.half()
    pipeline.unet.eval()

    # load scheduler
    pipeline.scheduler = DDIMScheduler.from_config("./pretrained_models/upscale_a_video/scheduler/scheduler_config.json")

    # load propagator
    if not args.propagation_steps == []:
        raft = RAFT_bi("./pretrained_models/upscale_a_video/propagator/raft-things.pth")
        propagator = Propagation(args.scale_factor, learnable=False)
    else:
        raft, propagator = None, None

    pipeline.propagator = propagator
    pipeline = pipeline.to(UAV_device)

    ## load LLaVA
    if use_llava:
        llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device, load_8bit=args.load_8bit_llava, load_4bit=False)
    else:
        llava_agent = None

    ## input
    if args.input_path.endswith(VIDEO_EXTENSIONS): # input a video
        video_list = [args.input_path]
    elif os.path.isdir(args.input_path) and \
         os.listdir(args.input_path)[0].endswith(IMAGE_EXTENSIONS): # input a image folder
        video_list = [args.input_path]
    elif os.path.isdir(args.input_path) and \
         os.listdir(args.input_path)[0].endswith(VIDEO_EXTENSIONS): # input a video folder
        video_list = get_video_paths(args.input_path)
    else:
        raise ValueError(f"Invalid input: '{args.input_path}' should be a path to a video file \
            or a folder containing videos.")

    ## ---------------------- start inferencing ----------------------
    for i, video_path in enumerate(video_list):
        vframes, fps, size, video_name = read_frame_from_videos(video_path)
        index_str = f'[{i+1}/{len(video_list)}]'
        print(f'{index_str} Processing video: ', video_name)

        if use_llava:
            print(f'{index_str} Generating video caption with LLaVA...')
            with torch.no_grad():
                video_img0 = vframes[0]
                w, h = video_img0.shape[-1], video_img0.shape[-2]
                fix_resize = 512
                _upsacle = fix_resize / min(w, h)
                w *= _upsacle
                h *= _upsacle
                w0, h0 = round(w), round(h)
                video_img0 = F.interpolate(video_img0.unsqueeze(0).float(), size=(h0, w0), mode='bicubic')
                video_img0 = (video_img0.squeeze(0).permute(1, 2, 0)).cpu().numpy().clip(0, 255).astype(np.uint8)
                video_img0 = Image.fromarray(video_img0)
                video_caption = llava_agent.gen_image_caption([video_img0])[0]

            wrapped_caption = textwrap.indent(textwrap.fill('Caption: '+video_caption, width=80), ' ' * 8)
            print(wrapped_caption)
        else:
            video_caption = ''

        prompt = video_caption + args.a_prompt

        vframes = (vframes/255. - 0.5) * 2 # T C H W [-1, 1]
        vframes = vframes.to(UAV_device)

        h, w = vframes.shape[-2:]
        if h>=1280 and w>=1280:
            vframes = F.interpolate(vframes, (int(h//4), int(w//4)), mode='area')

        vframes = vframes.unsqueeze(dim=0) # 1 T C H W
        vframes = rearrange(vframes, 'b t c h w -> b c t h w').contiguous()  # 1 C T H W

        if raft is not None:
            flows_forward, flows_backward = raft.forward_slicing(vframes)
            flows_bi=[flows_forward, flows_backward]
        else:
            flows_bi=None

        b, c, t, h, w = vframes.shape
        generator = torch.Generator(device=UAV_device).manual_seed(10)

        
        # For large resolution
        if h * w >= 384*384:
            args.perform_tile = True

        # ---------- Tile ----------
        torch.cuda.synchronize()
        start_time = time.time()
        if args.perform_tile:
            # 确保tile尺寸与缩放因子匹配
            tile_height = tile_width = args.tile_size
            # 增加重叠区域以解决缝隙问题
            tile_overlap_height = tile_overlap_width = max(64, int(128 * (args.scale_factor / 4)))
            output_h = h * args.scale_factor
            output_w = w * args.scale_factor
            output_shape = (b, c, t, output_h, output_w)  
            # 使用全0初始化输出
            output = vframes.new_zeros(output_shape)
            
            # 重新计算tile数量，确保覆盖整个图像
            effective_tile_width = tile_width - 2 * (tile_overlap_width // args.scale_factor)
            effective_tile_height = tile_height - 2 * (tile_overlap_height // args.scale_factor)
            
            tiles_x = max(1, math.ceil(w / effective_tile_width))
            tiles_y = max(1, math.ceil(h / effective_tile_height))
            print(f'{index_str} Processing the video w/ tile patches [{tiles_x}x{tiles_y}]...')  

            # 创建融合权重mask，用于平滑拼接边界
            def get_blend_mask(height, width):
                mask = torch.ones((1, 1, 1, height, width), device=UAV_device)
                # 用正弦函数在边缘创建过渡
                blend_width = min(tile_overlap_width, width // 4)
                blend_height = min(tile_overlap_height, height // 4)
                
                # 创建水平和垂直过渡区域
                if blend_width > 0:
                    for x in range(blend_width):
                        mask[:, :, :, :, x] *= math.sin(0.5 * math.pi * x / blend_width)
                        mask[:, :, :, :, width-x-1] *= math.sin(0.5 * math.pi * x / blend_width)
                
                if blend_height > 0:
                    for y in range(blend_height):
                        mask[:, :, :, y, :] *= math.sin(0.5 * math.pi * y / blend_height)
                        mask[:, :, :, height-y-1, :] *= math.sin(0.5 * math.pi * y / blend_height)
                        
                return mask

            # 创建用于累积权重的tensor
            weights_sum = torch.zeros_like(output)

            rm_end_pad_w, rm_end_pad_h = True, True
            if (tiles_x - 1) * effective_tile_width >= w:
                tiles_x = max(1, tiles_x - 1)
                rm_end_pad_w = False
                
            if (tiles_y - 1) * effective_tile_height >= h:
                tiles_y = max(1, tiles_y - 1)
                rm_end_pad_h = False

            # loop over all tiles
            for y in range(tiles_y):
                for x in range(tiles_x):
                    print(f"\ttile: [{y+1}/{tiles_y}] x [{x+1}/{tiles_x}]")
                    # 计算tile位置，确保最后一个tile覆盖到图像边缘
                    if x == tiles_x - 1 and not rm_end_pad_w:
                        ofs_x = w - tile_width
                        ofs_x = max(0, ofs_x)
                    else:
                        ofs_x = x * effective_tile_width
                    
                    if y == tiles_y - 1 and not rm_end_pad_h:
                        ofs_y = h - tile_height
                        ofs_y = max(0, ofs_y)
                    else:
                        ofs_y = y * effective_tile_height
                    
                    # 确保不会超出图像边界
                    input_start_x = ofs_x
                    input_end_x = min(ofs_x + tile_width, w)
                    input_start_y = ofs_y
                    input_end_y = min(ofs_y + tile_height, h)
                    
                    # 提取tile
                    input_tile = vframes[:, :, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    if flows_bi is not None:
                        flows_bi_tile = [
                            flows_bi[0][:, :, :, input_start_y:input_end_y, input_start_x:input_end_x],
                            flows_bi[1][:, :, :, input_start_y:input_end_y, input_start_x:input_end_x]
                        ]
                    else:
                        flows_bi_tile = None
                    
                    # 处理边缘情况，如果tile太小，则调整大小
                    if input_tile.shape[-1] < 32 or input_tile.shape[-2] < 32:
                        continue
                    
                    # 处理tile
                    try:
                        with torch.no_grad():
                            output_tile = pipeline(
                                prompt,
                                image=input_tile,
                                flows_bi=flows_bi_tile,
                                generator=generator,
                                num_inference_steps=args.inference_steps,
                                guidance_scale=args.guidance_scale,
                                noise_level=args.noise_level,
                                negative_prompt=args.n_prompt,
                                propagation_steps=args.propagation_steps,
                            ).images # C T H W [-1, 1]
                    except RuntimeError as error:
                        print('Error', error)
                        continue

                    # 计算输出tile在整个输出图像中的位置
                    out_start_x = input_start_x * args.scale_factor
                    out_end_x = input_end_x * args.scale_factor
                    out_start_y = input_start_y * args.scale_factor
                    out_end_y = input_end_y * args.scale_factor
                    
                    # 创建融合mask
                    mask = get_blend_mask(out_end_y - out_start_y, out_end_x - out_start_x)
                    
                    # 将输出tile融合到最终输出中
                    output[:, :, :, out_start_y:out_end_y, out_start_x:out_end_x] += output_tile * mask
                    weights_sum[:, :, :, out_start_y:out_end_y, out_start_x:out_end_x] += mask
            
            # 避免除零错误
            weights_sum = weights_sum.clamp(min=1e-8)
            
            # 根据权重对输出进行标准化
            output = output / weights_sum
        else:
            print(f'{index_str} Processing the video w/o tile...')
            try:
                with torch.no_grad():
                    output = pipeline(
                        prompt,
                        image=vframes,
                        flows_bi=flows_bi,
                        generator=generator,
                        num_inference_steps=args.inference_steps,
                        guidance_scale=args.guidance_scale,
                        noise_level=args.noise_level,
                        negative_prompt=args.n_prompt,
                        propagation_steps=args.propagation_steps,
                    ).images # C T H W [-1, 1]
            except RuntimeError as error:
                print('Error', error)

        # color correction
        if args.color_fix in ['AdaIn', 'Wavelet']:
            vframes = rearrange(vframes.squeeze(0), 'c t h w -> t c h w').contiguous()
            output = rearrange(output.squeeze(0), 'c t h w -> t c h w').contiguous()
            vframes = F.interpolate(vframes, scale_factor=args.scale_factor, mode='bicubic')
            if args.color_fix == 'AdaIn':
                output = adaptive_instance_normalization(output, vframes)
            elif args.color_fix == 'Wavelet':
                output = wavelet_reconstruction(output, vframes)
        else:
            output = rearrange(output.squeeze(0), 'c t h w -> t c h w').contiguous()

        output = output.cpu()

        torch.cuda.synchronize()
        run_time = time.time() - start_time

        ## ---------------------- saving output ----------------------
        prop = '_p' + '_'.join(map(str, args.propagation_steps)) if not args.propagation_steps == [] else ''
        suffix = '_' + args.save_suffix if not args.save_suffix == '' else ''
        save_name = f"{video_name}_n{args.noise_level}_g{args.guidance_scale}_s{args.inference_steps}_x{args.scale_factor}{prop}{suffix}"
        # save image
        if args.save_image:
            save_img_root = os.path.join(args.output_path, 'frame')
            save_img_path = f"{save_img_root}/{save_name}"
            os.makedirs(save_img_path, exist_ok=True)
            for i in range(output.shape[2]):
                save_image(output[i], f"{save_img_path}/{str(i).zfill(4)}.png", 
                normalize=True, value_range=(-1, 1))

        # save video
        save_video_root = os.path.join(args.output_path, 'video')
        os.makedirs(save_video_root, exist_ok=True)
        save_video_path = f"{save_video_root}/{save_name}.mp4"
        upscaled_video = (output / 2 + 0.5).clamp(0, 1) * 255
        upscaled_video = rearrange(upscaled_video, 't c h w -> t h w c').contiguous()
        upscaled_video = upscaled_video.cpu().numpy().astype(np.uint8)
        imageio.mimwrite(save_video_path, upscaled_video, fps=fps, quality=8, output_params=["-loglevel", "error"]) # Highest quality is 10, lowest is 0
        print(f'{index_str} Saving upscaled video... time (sec): {run_time:.2f} \n')

    print(f'\nAll video results are saved in {save_video_path}')