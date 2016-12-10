import os
import numpy as np
from neon.data import DataLoader, AudioParams
from neon.util.argparser import NeonArgparser

parser = NeonArgparser(__doc__)
args = parser.parse_args()

train_dir = os.path.join('/home/auto-114/PycharmProjects/neon_study_04/data', 'train')

noise_idx = '/home/auto-114/PycharmProjects/neon_study_04/data/noise-index.csv'
train_idx = '/home/auto-114/PycharmProjects/neon_study_04/data/train-index.csv'
val_idx = '/home/auto-114/PycharmProjects/neon_study_04/data/val-index.csv'

common_params = dict(sampling_freq=2000, clip_duration=2000, frame_duration=64, overlap_percent=50)
train_params = AudioParams(noise_index_file=noise_idx, noise_dir=train_dir, **common_params)
test_params = AudioParams(**common_params)
common = dict(target_size=1, nclasses=2)

# Validate...
# train = DataLoader(set_name='train', repo_dir=train_dir, media_params=train_params,
#                    index_file=train_idx, **common)
test = DataLoader(set_name='val', repo_dir=train_dir, media_params=test_params,
                  index_file=val_idx, **common)
print test.datum_size

