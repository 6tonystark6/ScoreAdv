import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import imageio
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import resnet50, vgg19, inception_v3
from pytorch_grad_cam import GradCAM, ScoreCAM, HiResCAM, GradCAMPlusPlus, AblationCAM, XGradCAM , LayerCAM, FullGrad, EigenCAM
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from timm import create_model
from torchvision.utils import save_image
from torch.utils.data import Subset, Dataset
from torchvision import transforms

from fr_model import IRSE_50, MobileFaceNet, IR_152, InceptionResnetV1
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


class base_dataset(Dataset):
    def __init__(self, dir, transform=None) -> None:
        super().__init__()

        self.dir = dir
        self.img_names = sorted(os.listdir(os.path.join(dir, 'src')))
        self.tgt_img_names = sorted(os.listdir(os.path.join(dir, 'target')))

        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, 'src', self.img_names[index])
        img = Image.open(img_path)

        tgt_index = index % len(self.tgt_img_names)
        tgt_img_path = os.path.join(self.dir, 'target', self.tgt_img_names[tgt_index])
        tgt_img = Image.open(tgt_img_path)

        if self.transform:
            img = self.transform(img)
            tgt_img = self.transform(tgt_img)

        return img, tgt_img

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    cam_model = resnet50(pretrained=True)
    target_layers = [cam_model.layer4[-1]]

    ir_model = IR_152([112, 112])
    ir_model.load_state_dict(th.load('pretrained_model/ir152.pth', map_location=th.device('cpu')))
    ir_model.eval().to(dist_util.dev())
    mobileface_model = MobileFaceNet(512)
    mobileface_model.load_state_dict(th.load('pretrained_model/mobile_face.pth', map_location=th.device('cpu')))
    mobileface_model.eval().to(dist_util.dev())
    facenet_model = InceptionResnetV1(num_classes=8631)
    facenet_model.load_state_dict(th.load('pretrained_model/facenet.pth', map_location=th.device('cpu')))
    facenet_model.eval().to(dist_util.dev())

    if args.vic_model == 'IR152':
        vic_model = ir_model
    elif args.vic_model == 'MobileFace':
        vic_model = mobileface_model
    elif args.vic_model == 'FaceNet':
        vic_model = facenet_model
    else:
        raise ValueError(f'Invalid model name: {args.vic_model}')

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []

    os.makedirs("samples", exist_ok=True)

    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])

    if args.dataset == 'celeba':
        dataset = base_dataset(dir='./celeba-hq_sample', transform=transform)
    elif args.dataset == 'lfw':
        dataset = base_dataset(dir='./lfw_sample', transform=transform)
    dataset = Subset(dataset, [x for x in range(args.num)])
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with ScoreCAM(model=cam_model, target_layers=target_layers) as cam:
        ASR = 0
        ASR_ir, ASR_facenet, ASR_mobileface = 0,0,0
        for i, (image, tgt_image) in enumerate(dataloader):
            print(f'start sampling number {i}...')
            model_kwargs = {"y": th.tensor([982] * args.batch_size, device=dist_util.dev())}
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )

            reference = image.float().float().contiguous()
            reference.requires_grad_()

            tgt_image = tgt_image.to(dist_util.dev())

            targets = [ClassifierOutputTarget(982)]
            grayscale_cam = cam(input_tensor=reference, targets=targets)
            score_cam = grayscale_cam[0, :]

            sample, asr, asr_ir, asr_facenet, asr_mobileface = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                reference=reference,
                score_cam=score_cam,
                model_kwargs=model_kwargs,
                clip_denoised=args.clip_denoised,
                tar_image=tgt_image,
                cond_fn=cond_fn,
                device=dist_util.dev(),
                progress=True,
                vic_model=vic_model,
                vic_name=args.vic_model,
                ir_model=ir_model,
                mobileface_model=mobileface_model,
                facenet_model=facenet_model,
                N=args.N,
                s_a=args.s_a,
                s_n=args.s_n
            )
            ASR += asr
            ASR_ir += asr_ir
            ASR_facenet += asr_facenet
            ASR_mobileface += asr_mobileface
            print('----->', 'ASR:', ASR, 'ASR_ir:', ASR_ir, 'ASR_facenet:', ASR_facenet, 'ASR_mobileface:', ASR_mobileface)

            os.makedirs(args.output_path, exist_ok=True)
            save_image(sample, f'{args.output_path}/{i}.png')

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()[0]

            all_images.append(sample)
            all_labels.append(i)
            logger.log(f"Generated image for class {i}")

        dist.barrier()
        logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1000,
        batch_size=1,
        timestep_respacing=250,
        use_ddim=False,
        model_path="models/256x256_diffusion.pt",
        classifier_path="models/256x256_classifier.pt",
        classifier_scale=1.0,
        reference_image="./reference_image",
        output_path='./FaceNet_scoreadv_lfw_outputs_N_2_sa_0.8_sn_0.6',
        vic_model='FaceNet',  # IR152, FaceNet, MobileFace
        dataset='lfw',  # celeba, lfw
        num=100,
        N=2,
        s_a=0.8,
        s_n=0.6
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
