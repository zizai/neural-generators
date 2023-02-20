import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter


class BaseTrainer(object):
    def __init__(self,
                 config,
                 save_video: bool = False):
        self.config = config

        data_dir = os.environ.get('NG_DATA_PATH') or os.path.expanduser('~/ng_data')
        time_start = datetime.now().__format__('%Y-%m-%d_%H-%M-%S')
        save_dir = os.path.join(data_dir, config.name, time_start)
        save_dir = Path(save_dir)
        print(save_dir)

        self.summary_writer = SummaryWriter(save_dir / 'tb')

        if save_video:
            self.video_folder = save_dir / 'video' / 'eval'
        else:
            self.video_folder = None

        checkpoint_dir = save_dir / 'checkpoint'
        checkpoint_dir.mkdir(exist_ok=True)

        np.random.seed(config.seed)
        random.seed(config.seed)

        self.name = config.name
        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir
        self.seed = config.seed

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        raise NotImplementedError
