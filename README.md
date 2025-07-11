# ScoreAdv: Score-based Targeted Generation of Natural Adverarial Examples via Diffusion Models
By Chihan Huang, Hao Tang

Despite the remarkable success of deep learning across various domains, these models remain vulnerable to adversarial attacks. Although many existing adversarial attack methods achieve high success rates, they typically rely on $\ell_{p}$-norm perturbation constraints, which do not align with human perceptual capabilities. Consequently, researchers have shifted their focus toward generating natural, unrestricted adversarial examples (UAEs). Traditional approaches using GANs suffer from inherent limitations, such as poor image quality due to the instability and mode collapse of GANs. Meanwhile, diffusion models have been employed for UAE generation, but they still predominantly rely on iterative PGD perturbation injection, without fully leveraging the denoising capabilities that are central to the diffusion model. In this paper, we introduce a novel approach for generating UAEs based on diffusion models, named ScoreAdv. This method incorporates an interpretable adversarial guidance mechanism to gradually shift the sampling distribution towards the adversarial distribution, while using an interpretable saliency map technique to inject the visual information of a reference image into the generated samples. Notably, our method is capable of generating an unlimited number of natural adversarial examples and can attack not only image classification models but also image recognition and retrieval models. We conduct extensive experiments on the ImageNet and CelebA datasets, validating the performance of ScoreAdv across ten target models in both black-box and white-box settings. Our results demonstrate that ScoreAdv achieves state-of-the-art attack success rates and image quality. Furthermore, due to the dynamic interplay between denoising and adding adversarial perturbation in the diffusion model, ScoreAdv maintains high performance even when confronted with defense mechanisms, showcasing its robustness.

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

## Overall Framework

![image](https://github.com/6tonystark6/ScoreAdv/blob/main/images/overall%20framework.png)

## Code Organization
```
project-root/
├── celeba-hq_sample/          # place to put celeba-hq dataset files
│   ├── src/                   # put source images
│   └── target/                # put target images
├── fr_model/                  # attacked face recognition models
│   ├── __init__.py            # init file
│   ├── facenet.py             # code for facenet
│   ├── ir152.py               # code for ir152
│   └── irse.py                # code for irse
├── guided_diffusion/          # main code for guided diffusion
│   ├── __init__.py            # init file
│   ├── dist_util.py           # code for distributed training
│   ├── fp16_util.py           # enabling half-precision
│   ├── gaussian_diffusion.py  # main code for scoreadv attack
│   ├── image_datasets.py      # dataset loading and preprocessing
│   ├── logger.py              # logger
│   ├── losses.py              # code for various loss functions
│   ├── nn.py                  # neural net components
│   ├── resample.py            # resampling strategy
│   ├── respace.py             # timestep schedule reparameterization
│   ├── script_util.py         # utility functions
│   ├── train_util.py          # training loop functions
│   └── unet.py                # core U-Net architecture
├── lfw_sample/                # place to put lfw dataset files
│   ├── src/                   # put source images
│   └── target/                # put target images
├── models/                    # place to put trained diffusion model checkpoints
├── pretrained_model/          # place to put trained fr model checkpoints
├── reference_image/           # place to put reference images
│   ├── images/                # place to put reference images
│   └── labels.csv             # place to put reference image labels
├── get_lfw_image.py           # select lfw images for evaluation
├── model-card.md              # instructions for training diffusion model
├── README.md                  # readme
└── scoreadv_rec.py            # main code for scoreadv evaluation
```

# Requirements

- python == 3.9.21
- pytorch == 2.0.0
- torchvision == 0.15.0
- torchaudio == 2.0.0
- torchattacks == 3.5.1
- torch-fidelity == 0.3.0
- timm == 1.0.15
- numpy == 1.24.3
- mpi4py == 3.1.4
- matplotlib == 3.9.4
- lpips == 0.1.4
- scipy == 1.13.1
- tqdm == 4.67.1
- pandas == 2.0.3
- PyYAML == 6.0.2
- imageio == 2.37.0
- triton == 2.0.0
- huggingface-hub == 0.29.2
- grad-cam == 1.5.4
- fsspec == 2025.3.0

## Use

### Download pre-trained models

OpenAI have released checkpoints for the main diffusion models in the paper. Before using these models, please review the corresponding [model card](model-card.md) to understand the intended use and limitations of these models.

Here are the download links for each model checkpoint:

 * 64x64 classifier: [64x64_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt)
 * 64x64 diffusion: [64x64_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt)
 * 128x128 classifier: [128x128_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_classifier.pt)
 * 128x128 diffusion: [128x128_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt)
 * 256x256 classifier: [256x256_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt)
 * 256x256 diffusion: [256x256_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt)
 * 256x256 diffusion (not class conditional): [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)
 * 512x512 classifier: [512x512_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_classifier.pt)
 * 512x512 diffusion: [512x512_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt)

### Datasets

The datasets used in face recognition attack are [celeba-hq](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [lfw](https://vis-www.cs.umass.edu/lfw/), which can be downloaded and placed in the corresponding folder.

### Evaluate attack performance

To sample from these models, you can use the `scoreadv_rec.py` scripts. Here, we provide flags for sampling from all of these models. We assume that you have downloaded the relevant model checkpoints into a folder called `models/`.

For the following example, we will generate 1000 samples with batch size 1.

```Python
SAMPLE_FLAGS="--batch_size 1 --num_samples 1000 --timestep_respacing 250"
```

You can also adjust the `classifier_scale, reference_image, output_path, vic_model, dataset, N, s_a and s_n` in `create_argparser()` in `scoreadv_rec.py` according to your need.

```Python
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python scoreadv_rec.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS
```

And the attacked images will be saved in the `output_path` directory, attack performance will be printed directly.

### Train diffusion model yourself

Training diffusion models is described in the [openai/improved-diffusion](https://github.com/openai/improved-diffusion). Training a classifier is similar. We assume you have put training hyperparameters into a `TRAIN_FLAGS` variable, and classifier hyperparameters into a `CLASSIFIER_FLAGS` variable. Then you can run:

```Python
mpiexec -n N python scripts/classifier_train.py --data_dir path/to/imagenet $TRAIN_FLAGS $CLASSIFIER_FLAGS
```

Here are flags for training the 128x128 classifier. You can modify these for training classifiers at other resolutions:

```Python
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
```

For sampling from a 128x128 classifier-guided model, 25 step DDIM:

```Python
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 128 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 50000 --timestep_respacing ddim25 --use_ddim True"
mpiexec -n N python scripts/classifier_sample.py \
    --model_path /path/to/model.pt \
    --classifier_path path/to/classifier.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS
```

To sample for 250 timesteps without DDIM, replace `--timestep_respacing ddim25` to `--timestep_respacing 250`, and replace `--use_ddim` True with `--use_ddim` False.

### Image classification attack

For image classification attack, you can download the code [here](https://pan.baidu.com/s/1afMpksYli_PIi1yJGxoiUw?pwd=ghiw) and follow the `README.md` to perform attack.

## Citation
```
@misc{huang2025scoreadvscorebasedtargetedgeneration,
      title={ScoreAdv: Score-based Targeted Generation of Natural Adversarial Examples via Diffusion Models}, 
      author={Chihan Huang and Hao Tang},
      year={2025},
      eprint={2507.06078},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.06078}, 
}
```
