# SoundBrush (AAAI 2025)

### [Project Page](https://soundbrush.github.io/) | [Paper](https://arxiv.org/abs/2501.00645) | [Dataset](https://soundbrush.github.io/)
This repository contains a pytorch implementation for the AAAI 2025 paper, [SoundBrush: Sound as a Brush for Visual Scene Editing](https://soundbrush.github.io/). SoundBrush can manipulate scenes to reflect the mood of the input audio or to insert sounding objects while preserving the original content.<br><br>

<img width="800" alt="teaser" src="./assets/teaser.png"> 

## Getting started
This code was developed on Ubuntu 18.04 with Python 3.8, CUDA 11.7 and PyTorch 2.0.1. Later versions should work, but have not been tested.

### Installation
Create and activate a virtual environment to work in:
```
conda create --n soundbrush python=3.8
conda activate soundbrush
```

Install the requirements with pip and [PyTorch](https://pytorch.org/). For CUDA 11.7, this would look like:
```
pip insall -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

### Download models
To run SoundBrush, you need to download the pretrained model.
Download [pretrained model](https://drive.google.com/file/d/1W3W34L-ERt_n4mm7Osx9-6IXkbanOWdF/view?usp=sharing).

After downloading the models, place them in `./checkpoints`.
```
./checkpoints/model.ckpt
```

### Download dataset
The dataset involves a fully synthetic subset, consisting of generated image pairs with corresponding audio, and a real-data-involved subset, which is synthetically generated but incorporates real images and audio.
Download [train dataset](https://drive.google.com/file/d/1YnqwoHzHv3kC0JcD7N3KXmuJGRIYcMbd/view?usp=sharing) and [test dataset](https://drive.google.com/file/d/1KKmJ0Fq1u1Xom-iTdil0-3qeRI0kJZ09/view?usp=sharing).


## Demo
Run below command to inference the model.
We provide sample images and audios in **./source_images** and **./source_wavs**, respectively. 
The edited images will be saved in **./outputs**

```
python edit_inference.py --audio_dir <audio directory> --img_dir <ikmg dir> --save_dir <output dir>

#or simply run

sh inference.sh
```

## Agreement
- The SoundBrush dataset is provided for non-commercial research purposes only. 
- All wavfile and images of the SoundBrush dataset are sourced from the Internet and do not belong to our institutions. Our institutions do not take responsibility for the content or the meaning of these videos.
- You agree not to reproduce, duplicate, copy, sell, trade, resell, or exploit any portion of the videos and any portion of derived data for commercial purposes.
- You agree not to further copy, publish, or distribute any portion of the SoundBrush dataset. Except, it is allowed to make copies of the dataset for internal use at a single site within the same organization.


## **Notes**
```
@inproceedings{soundbrush,
  title     = {SoundBrush: Sound as a Brush for Visual Scene Editing},
  author    = {Sung-Bin, Kim and Jun-Seong, Kim and Ko, Junseok and Kim, Yewon and Oh, Tae-Hyun},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
}
```


## **Acknowledgement**
We heavily borrow the code from [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) and [ImageBind](https://github.com/facebookresearch/ImageBind) and the dataset from [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/), and the agreement statement from [CelebV-HQ](https://github.com/CelebV-HQ/CelebV-HQ?tab=readme-ov-file). We sincerely appreciate those authors.
