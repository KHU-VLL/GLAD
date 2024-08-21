# GLAD: Global-Local View Alignment and Background Debiasing for Unsupervised Video Domain Adaptation with Large Domain Gap [WACV 2024]
![method](resources/method.jpg)

<p>
    <span>
        <a href="https://arxiv.org/abs/2311.12467">arXiv</a> | 
        <a href="https://openaccess.thecvf.com/content/WACV2024/html/Lee_GLAD_Global-Local_View_Alignment_and_Background_Debiasing_for_Unsupervised_Video_WACV_2024_paper">paper</a>
    </span>
</p>


## What is GLAD?
In this work, we tackle the challenging problem of unsupervised video domain adaptation (UVDA) for action recognition.
We specifically focus on scenarios with **a substantial domain gap**, in contrast to existing works primarily deal
with small domain gaps between labeled source domains and unlabeled target domains.

So, contributions of this work is 2-fold.

### 1. Introduces Kinetics→BABEL.
To establish a more realistic setting, we introduce a novel UVDA scenario, denoted as **Kinetics→BABEL**,
with a more considerable domain gap in terms of both temporal dynamics and background shifts.

### 2. Introduces a method to tackle the challenging Kinetics→BABEL.
- To tackle the temporal shift, i.e., action duration difference between the source and target domains,
we propose a global-local view alignment approach.
- To mitigate the background shift, we propose to learn temporal order sensitive representations by temporal order
learning and background invariant representations by background augmentation. We empirically validate that the proposed method
shows significant improvement over the existing methods on the Kinetics→BABEL dataset with a large domain gap.

## Installation
We provide our working conda environment as an exported yaml file.
```bash
conda env create --file requirements/environment.yml
pip install -e .
```

## Data Preparation

### 1. Download Kinetics→BABEL.

Kinetics→BABEL consists of 4 txt files `babel_test.txt`, `babel_train.txt`, `k400_test.txt`, and `k400_train.txt`.
The size of these in total is 770 KB. 

### 2. Download AMASS BMLrub Rendered Videos.

The AMASS dataset is a comprehensive motion capture skeleton dataset that serves as an input for the [BABEL](https://babel.is.tue.mpg.de/index.html) dataset.
Unlike the original, our proposed dataset, Kinetics→BABEL, utilizes a different kind of input—rendered videos rather than skeletons.
To acquire an access to them, please create an account on [AMASS](https://amass.is.tue.mpg.de/) and download the BMLrub rendered videos.

### 3. Link datasets.

Make symlinks to the actual dataset paths.
```bash
mkdir data
ln -s ./data/k400 /KINETICS/PATH/
ln -s ./data/babel /BABEL/PATH/
```
We highly recommend to extract rawframes beforehand to optimize I/O.
Below are example structures for each dataset.

<details><summary>Kinetics Structure</summary>

```
./data/k400/rawframes_resized
├── train
│   ├── applauding
│   │   ├── 0nd-Gc3HkmU_000019_000029
│   │   │   ├── img_00000.jpg
│   │   │   ├── img_00001.jpg
│   │   │   ├── img_00002.jpg
│   │   │   └── ...
│   │   ├── 0Tq8uFakTbk_000000_000010
│   │   ├── 0XrsfW9ejfk_000000_000010
│   │   ├── 0YQrMye3BBY_000000_000010
│   │   ├── 1WMulo84kBY_000020_000030
│   │   └── ...
│   ├── balloon_blowing
│   ├── ...
│   ├── unboxing
│   └── waxing_legs
└── val
    ├── applauding
    ├── balloon_blowing
    ├── ...
    ├── unboxing
    └── waxing_legs
```
</details>

<details><summary>BABEL Structure</summary>

```
./data/babel
├── train
│   ├── 000000
│   │   ├── img_00001.jpg
│   │   ├── img_00002.jpg
│   │   └── ...
│   ├── 000002
│   └── ...
└── val
    ├── ...
    ├── 013286
    └── 013288
```
</details>

### 3. Extract Backgrounds for the Background Augmentation.
```bash
python utils/extract_median_by_rawframes.py \
    --ann-file 'data/filelists/k400/filelist_k400_train_closed.txt' \
    --outdir 'data/median/k400' \
    --start-index 0 \
    --data-prefix 'data/k400/rawframes_resized'
```


## Train and Test

### Train
The training process has 2 stages.

1. Pretrain TOL (Temporal Ordering Learning)
    ```bash
    source tools/dist_train.sh configs/tol.py 8 \
    --seed 0
    ```
    Then training result will be generated under `work_dirs/tol/`, which will be utilized in the next stage.
2. GLAD
    ```bash
    source tools/dist_train.sh configs/glad.py 8 \
    --seed 3 \
    --validate --test-last --test-best
    ```

### Test

```bash
source tools/dist_test.sh configs/glad.py $(find 'work_dirs/glad' -name '*best*.pth' | head -1) 8 \
--eval 'mean_class_accuracy' 'confusion_matrix'
```

## Special Thanks
This project has been made possible through the generous funding and support of NCSOFT Corporation. We extend our sincere gratitude for their contribution and belief in our work.


## License
This project is released under the [BSD-3-Clause](LICENSE).


## Citation

Thank you for cosidering to cite this work!!!
It would be so GLAD to help you using the dataset Kinetics→BABEL as well as this code base.
So, please leave an issue or send an email to us whenever having problems.

Please cite by including this bibtex.

Hope you enjoy your research!

```bibtex
@inproceedings{leebae2024glad,
  title={{GLAD}: Global-Local View Alignment and Background Debiasing for Video Domain Adaptation},
  author={Lee, Hyogun and Bae, Kyungho and Ha, Seong Jong and Ko, Yumin and Park, Gyeong-Moon and Choi, Jinwoo},
  booktitle={Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```
