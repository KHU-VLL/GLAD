from pathlib import Path
import os
import re

from pprint import pprint
import cv2
import numpy as np
from einops import rearrange

from moviepy.editor import *

from ..builder import PIPELINES
from slurm.utils.commons.labelmaps import labelmaps


FPS = 30
DURATION = 5
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


@PIPELINES.register_module()
class DebugInterPipelineVisualizer:
    def __init__(self, stop=False, namespace='clip_viz', ext='jpg'):
        """An example of use case

        pipelines = [
            ...,
            dict(type='DebugInterPipelineVisualizer', namespace='clip_viz/01_before', stop=False),
            dict(type='DebugInterPipelineVisualizer', namespace='clip_viz/cop', stop=4),
            dict(type='DebugInterPipelineVisualizer', namespace='clip_viz/cop/mp4', stop=4, ext='mp4'),
            ...
            dict(type='DebugInterPipelineVisualizer', namespace='clip_viz/02_after', stop=True),
            ...
        ]
        """
        self.stop = stop
        self.stop_counter = 1
        self.namespace = namespace
        assert ext in ['jpg', 'png', 'mp4', 'gif']
        self.ext = ext

    def __call__(self, results):
        assert 'imgs' in results
        imgs = results['imgs']  # (N x T) x [H, W, C], no batch size: each worker processes a single data
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        assert len(imgs) == num_clips * clip_len, f'{len(imgs)} != {num_clips} x {clip_len}'
        print(f'\n\nimage shape: ({num_clips} x {clip_len} = {len(imgs)}) x {imgs[0].shape}')

        results_tmp = results.copy()
        del results_tmp['imgs']
        pattern = r'(/data\d?)?/local_datasets/'
        if 'filename' in results_tmp:
            path = re.sub(pattern, './data/', results_tmp['filename'])
            results_tmp['filename'] = path
        elif 'frame_dir' in results_tmp:
            path = re.sub(pattern, './data/', results_tmp['frame_dir'])
            results_tmp['frame_dir'] = path
        dataset_name, = re.findall(r'(babel|ucf|hmdb|epic|k400)', path)

        p_tmp = Path(f'tmp/{self.namespace}'); p_tmp.mkdir(parents=True, exist_ok=True)
        imgs = np.array(imgs)  # [N x T, H, W, C]
        p_clip = p_tmp / f'{os.getpid()}_{self.stop_counter:02d}.{self.ext}'
        frame_inds = results['frame_inds'].reshape(num_clips, clip_len)
        print(frame_inds)

        if self.ext in ['jpg', 'png']:
            imgs = rearrange(imgs, '(n t) h w c -> (n h) (t w) c', n=num_clips, t=clip_len)
            cv2.imwrite(str(p_clip), cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))

        elif self.ext in ['mp4', 'gif']:
            label = int(results['label'])
            labelname = labelmaps[dataset_name][label]
            if num_clips > 1 and clip_len == 1:  # uniform sampling => num_clips is T
                N, T = clip_len, num_clips
                clips = rearrange(imgs, '(n t) h w c -> n t h w c', n=N, t=T)
                frame_inds = frame_inds.reshape(-1)
                index_clips = [TextClip(f'idx: {",".join(list(map(str, frame_inds)))}', fontsize=12)]
                labelname_clip = TextClip(labelname, fontsize=24, bg_color='white').set_position('center').set_opacity(.3)
            else:
                N, T = num_clips, clip_len
                clips = rearrange(imgs, '(n t) h w c -> n t h w c', n=N, t=T)
                index_clips = [TextClip(f'idx: {idx[0]}~{idx[-1]}', fontsize=18) for idx in frame_inds]
                labelname_clip = TextClip(labelname, fontsize=100, color='white').set_position('center').set_opacity(.3)
            clips = [ImageSequenceClip([frame for frame in clip], fps=FPS) for clip in clips]  # the input should be in type list of ndarrays
            clips = [CompositeVideoClip([clip, index_clip]) for clip, index_clip in zip(clips, index_clips)]  # frame idxs at left top for each clip
            clips = [clip.margin(1) for clip in clips]
            clips = clips_array([clips])
            clips = CompositeVideoClip([clips, labelname_clip])
            clips = clips.set_duration(T / FPS)
            r = DURATION * FPS // T  # 5 sec
            clips = concatenate_videoclips([clips] * r)
            if self.ext == 'mp4':
                clips.write_videofile(str(p_clip), logger=None)
            elif self.ext == 'gif':
                clips.write_gif(str(p_clip), logger=None)

        print(f'\n\nStored the result in {p_clip}')

        print()
        print(imgs.shape)
        pprint(results_tmp)

        will_exit = False
        if isinstance(self.stop, int):
            if self.stop_counter == self.stop:
                will_exit = True
            else:
                self.stop_counter += 1
        elif self.stop:
            will_exit = True

        if will_exit:
            print('\n\nmmaction/datasets/pipelines/visualizer.py:DebugInterPipelineVisualizer stopped the process\n')
            exit()
        else:
            print('\n\n')
            return results
