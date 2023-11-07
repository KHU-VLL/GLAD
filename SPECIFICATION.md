# Outline
- The key parts of our method are the following three.
  - Global-local view alignment (GLAD
  - Background augmentation
  - Temporal Ordering Learning (TOL)

# GLA
## Implementations
### `mmaction/models/recognizers/pyramidic_recognizers.py`
```python
class FrameSampler:
    def __init__(self, sampler_name, sampler_index=dict()):
        self.sampler_name = sampler_name
        self.sampler = getattr(FrameSampler, self.sampler_name)
        assert isinstance(sampler_index, dict)
        self.index_dict = sampler_index
    def __call__(self, imgs):
        return self.sampler(imgs, self.index_dict)
    ...
```
- We initially take 8 of 8-framed local clips, which forms a tensor shaped `[8, 8, H, W, C]`.
- Then we can consider that tensor as a $8 \times 8$ matrix of $H \times W \times C$ images whose row indicates a local clip, and column indicates a global clip.
- `FrameSampler` samples global and local clips based on `sampler_index` given in the model config file.
- Values we used are `sampler_index=dict(l=[4, 5], g=3)`.

```python
@RECOGNIZERS.register_module()
class TemporallyPyramidicRecognizer(BaseRecognizer):
    def __init__(self,
                 sampler_name:str,
                 sampler_index=dict(),
                 dim='2d',
                 fuse_before_head=True,
                 consensus_before_head=True,
                 locality_aware_head=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler_name = sampler_name
        self.sampler = FrameSampler(self.sampler_name, sampler_index)
        self.dim = dim
        self.fuse_before_head = fuse_before_head
        self.consensus_before_head = consensus_before_head
        self.locality_aware_head = locality_aware_head
    ...
```
- `TemporallyPyramidicRecognizer` or a 'recognizer' in mmaction is a bundle of the whole structure of model: backbone, necks, head.
- All we did are aggregating `FrameSampler` forward of it and manipulating multiview tensors.
- Frames of videos are fed into this class and it outputs the loss.

### `mmaction/models/necks/domain_classifier.py`
```python
@NECKS.register_module()
class TemporallyPyramidicDomainClassifier(DomainClassifier):
    def __init__(self,
        temporal_locality,
        *args, **kwargs
    ):
        self.temporal_locality = temporal_locality
        super().__init__(*args, **kwargs)

    def forward(self,
        f_tallies:list,
        labels=None, domains=None,
        train=False,
        **kwargs
    ):
        ...
        if self.temporal_locality in ['local', 'global', 'both']:
            fs, domains = temporal_locality_fuse(
                f_tallies,
                self.temporal_locality,
                return_domain_names=True
            )
        elif self.temporal_locality == 'local-global':  # not used for this project
            ...
        elif self.temporal_locality == 'global-local':  # not used for this project
            ...
        elif self.temporal_locality == 'cross':  # not used for this project
            ...
        _, losses = super().forward(
            fs, domains=domains, train=train, **kwargs)
        return f_tallies, losses
```
- The GLADis a set of domain classifiers with individual views.
- A term `temporal_locality` is a legacy version of the term `view`.
- Each domain classifier has its own view `'local'`, `'global'` or `'both'`.
- And they are aggreagated in `TemporallyPyramidicRecognizer`.
- `f_tallies` is a list of lists of tensors, each of which indicates
  ```python
    f_tallies = [
        [source_local, source_global],
        [target_local, target_global]
    ]
  ```
- This module passes `f_tallies` to `temporal_locality_fusion` for manipulation.

### `mmaction/models/common/temporal_locality_fusion.py`
```python
def temporal_locality_fuse(
    f_tallies:List[List[torch.Tensor]],
    temporal_locality='both',
    fusion_method='mean',
    return_domain_names=False,
):
    elif temporal_locality in ['local', 'global']:
        if fusion_method == '':             ...
        elif fusion_method == 'concat':     ...
        elif fusion_method == 'mean':       ...
    elif temporal_locality == 'both':
        if fusion_method == '':             ...
        elif fusion_method == 'concat':     ...
        elif fusion_method == 'mean':       ...
```
- `temporal_locality_fuse` fuses `f_tallies` into a single tensor.
- A fusion method is selected in the config file and it is different among domain classifiers.
- We finally adopted `'mean'` method.


# Background augmentation
## `mmaction/datasets/pipelines/augmentations.py`, line 1916.
```python
@PIPELINES.register_module()
class BackgroundBlend:
    def __init__(
        self,
        io_backend='disk', decoding_backend='cv2',
        resize_h=256, crop_size=224,
        p=.5,  # probability to apply
        ann_files=[],
        data_prefixes=[],
        alpha=.5,  # blend ratio of origin image
        blend_label=False
    ):
        if np.random.rand() > self.p:  # w.p 1-p
            return results
        ...
        bg_img = self._preprocess_bg(bg_img)
        alpha = self.alpha if type(self.alpha) == float else np.random.rand()
        imgs = [alpha * img + (1 - alpha) * bg_img for img in imgs]
        ...
```
- `BackgroundBlend` is an augmentation which prior to a `FrameSampler`.
- It can be configured in the config file.
- `p` stands for a probability whether this augmentation will be applied.
- `alpha` is a blend ratio.
- We used an optimized fixed value of `p=0.25, alpha=0.75`.
- This is an example of configuration.
    ```python
    dict(
        type='BackgroundBlend',
        p=0.25,
        alpha=0.75,
        resize_h=256,
        crop_size=224,
        ann_files=[
            './data/filelists/k400/filelist_k400_train_closed.txt'
        ],
        data_prefixes=['./data/median/k400/train'],
        blend_label=False),
    ```
- `ann_files` may be provided or not.
  - If provided, the module will select a BG from a set of TMF of videos in the `ann_file`.
  - Else if `None`, the module will do so from a set of all `jpg` files under `data_prefixes`.


# Temporal Ordering Learning (TOL)
## `mmaction/models/heads/timesformer_cop_head.py`
- Please ignore `'timesformer'` in the name of the file.
- We just referred code from [the original paper](https://github.com/xudejing/video-clip-order-prediction/blob/master/models/vcopn.py).
- And edited as little as possible just enough for it to be running on mmaction.
