# Unifying Guided and Unguided Outdoor Image Synthesis

![result1](https://github.com/Usman-Rafique/Usman-Rafique.github.io/blob/069ac7b123f7628276b92af50acb5338911166c0/un_guided/guided.png)

## Using the Code
This repository contains scripts for training and generating visual results. Most of the settings are stored in the file `config.py`. Before running any training or visualization, please make sure that settings in `config.py` are correct. 

### Training
For training, you can specify a directory through `cfg.train.out_dir` (in `config.py`). Trained models and training logs will be saved in this directory. To train the model, use command `python3 train.py`.

### Visualization
For visual results, run `python3 visualize.py`.

## Permission
The code is released only for academic and research purposes.

## Recommended citation
<pre>
@inproceedings{unifying2021,
  title={Unifying Guided and Unguided Outdoor Image Synthesis},
  author={Rafique, Muhammad Usman and Zhang, Yu and Brodie, Benjamin and Jacobs, Nathan},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2021}}
</pre>
