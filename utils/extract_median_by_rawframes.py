from pathlib import Path
import numpy as np
import imageio
import argparse

from mmaction.datasets import RawframeDataset

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn)
from multiprocessing import Pool


def sample_all(results:dict):
    results['frame_inds'] = results['start_index'] + np.arange(results['total_frames'])
    return results


progress_args = [
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),  # not default
    TimeRemainingColumn(),
    MofNCompleteColumn()
]


def main():
    args = parse_args()
    dataset = build_dataset(args)
    num_videos = len(dataset.video_infos)
    with Pool() as pool, Progress(*progress_args) as progress:
        task = progress.add_task('\t\t[green]Processing...', total=num_videos)
        for done in pool.imap_unordered(worker, list(range(num_videos))):
            progress.update(task, advance=done)

def build_dataset(args):
    pipeline = [sample_all, dict(type='RawFrameDecode')]
    dataset = RawframeDataset(
        ann_file=str(args.ann_file),
        filename_tmpl=args.filename_tmpl,
        start_index=args.start_index,
        data_prefix=args.data_prefix,
        with_offset=args.with_offset,
        pipeline=pipeline,
    )
    return dataset


def worker(i):
    args = parse_args()
    dataset = build_dataset(args)
    results = dataset[i]
    frames = np.array(results['imgs'])  # [T, H, W, C]
    tmf = np.median(frames, axis=0).astype(np.uint8)
    p_root_outdir:Path = args.outdir  # Path('/tmp/median/k400_nec-drone/k400')
    p_rel_video = Path(results['frame_dir'].split(dataset.data_prefix)[-1][1:])  # train/shaking_hands/-0HRnFhCDdc_000026_000036
    p_tmf = (p_root_outdir / p_rel_video).with_suffix('.jpg')  # train/shaking_hands/-0HRnFhCDdc_000026_000036.jpg
    if not p_tmf.exists():
        # why exist_ok though already checked the existence?
        # => bc not thread-safe
        p_tmf.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(p_tmf), tmf)
    return True  # well done


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-file', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--filename-tmpl', type=str, default='img_{:05d}.jpg')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--data-prefix', type=str)
    parser.add_argument('--with-offset', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
