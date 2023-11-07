# from collections.abc import Sequence
from typing import Union, Dict, List
from math import isclose

import numpy as np

from ..builder import PIPELINES
from .compose import Compose


Transform = dict
ListOfTransforms = List[Transform]
MultiWayTransforms = Dict[str,ListOfTransforms]


@PIPELINES.register_module()
class Switch:
    def __init__(self, 
        transform_threads:MultiWayTransforms,  # list of dicts or list of lists of dicts
        betas:Dict[str,float],  # should be sum to 1
        alpha:float=.5,  # the proba that this aug is applied
    ):
        assert isinstance(transform_threads, dict)
        assert len(transform_threads) == len(betas)
        assert isclose(sum(betas.values()), 1)
        assert betas.keys() == transform_threads.keys()
        assert 0 <= alpha <= 1

        self.betas = betas
        self.alpha = alpha
        self.transform_threads = {}
        for thread_name, transforms in transform_threads.items():
            transforms = Compose(transforms)
            self.transform_threads[thread_name] = transforms

    def __call__(self, results):
        switch = np.random.rand() < self.alpha
        results['switch'] = switch
        if switch:  # apply
            keys, probas = zip(*self.betas.items())
            thread_name = np.random.choice(keys, p=probas)
            selected_thread = self.transform_threads[thread_name]
            results = selected_thread(results)
            results['switch_methods'] = repr(selected_thread)
        return results

    def __repr___(self):
        format_string = self.__class__.__name__ + '(\n'
        for thread in self.transform_threads:
            format_string += '    ['
            format_string += f'\n        {repr(thread)}'
            format_string += '\n    ]'
        format_string += '\n)'
        return format_string
