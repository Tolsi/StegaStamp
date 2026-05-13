## StegaStamp: Invisible Hyperlinks in Physical Photographs [[Project Page]](http://www.matthewtancik.com/stegastamp)

### CVPR 2020
**[Matthew Tancik](https://www.matthewtancik.com), [Ben Mildenhall](http://people.eecs.berkeley.edu/~bmild/), [Ren Ng](https://scholar.google.com/citations?hl=en&user=6H0mhLUAAAAJ)**
*University of California, Berkeley*

![](docs/teaser.png)

## Introduction
This repository is a code release for the ArXiv report found [here](https://arxiv.org/abs/1904.05343). The project explores hiding data in images while maintaining perceptual similarity. Our contribution is the ability to extract the data after the encoded image (StegaStamp) has been printed and photographed with a camera (these steps introduce image corruptions). This repository contains the code and pretrained models to replicate the results shown in the paper. Additionally, the repository contains the code necessary to train the encoder and decoder models.

## Citation
If you find our work useful, please consider citing:
```
    @inproceedings{2019stegastamp,
        title={StegaStamp: Invisible Hyperlinks in Physical Photographs},
        author={Tancik, Matthew and Mildenhall, Ben and Ng, Ren},
        booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2020}
    }
```

## Installation
- Clone repo
```bash
git clone https://github.com/Tolsi/StegaStamp.git
cd StegaStamp
```

- Install dependencies
```bash
pip install -r requirements.txt
```

- Extract pretrained models from archives
```bash
cd saved_models
cat stegastamp_pretrained.zip.* > stegastamp_pretrained.zip
unzip stegastamp_pretrained.zip
cd ../detector_models
cat stegastamp_detector.zip.* > stegastamp_detector.zip
unzip stegastamp_detector.zip
```

## Usage

### Encoding a Message
Encode a message into an image (max 7 characters / 56 bits):
```bash
python encode_image.py \
  saved_models/stegastamp_pretrained \
  --image test_im.png \
  --save_dir out/ \
  --secret Hello
```

### Decoding a Message
Decode a message from a StegaStamp:
```bash
python decode_image.py \
  saved_models/stegastamp_pretrained \
  --image out/test_im_hidden.png
```

### Detecting and Decoding
Detect and decode StegaStamps in a video:
```bash
python detector.py \
  --detector_model detector_models/stegastamp_detector \
  --decoder_model saved_models/stegastamp_pretrained \
  --video test_vid.mp4
```

## Training
Set dataset path in train.py:
```
TRAIN_PATH = DIR_OF_DATASET_IMAGES
```

Train model:
```bash
bash scripts/base.sh EXP_NAME
```

## Performance
- Encoding time: ~11 sec (CPU), ~6 sec (with GPU)
- Decoding time: ~11 sec (CPU)
- Message capacity: 7 characters (56 bits)

## Requirements
- Python 3.9+
- TensorFlow 2.17+
- See requirements.txt for full list
