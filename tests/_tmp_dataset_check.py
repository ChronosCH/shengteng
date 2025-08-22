import sys
from pathlib import Path

# Ensure training package import
sys.path.append(str(Path(__file__).resolve().parent.parent / 'training'))

from enhanced_cecsl_trainer import EnhancedCECSLConfig, EnhancedCECSLDataset


def main():
    cfg = EnhancedCECSLConfig()
    # Prefer CS-CSL; fallback handled inside dataset
    cfg.data_root = str(Path(__file__).resolve().parent.parent / 'data' / 'CS-CSL')

    ds_train = EnhancedCECSLDataset(cfg, 'train', use_augmentation=False)
    ds_dev = EnhancedCECSLDataset(cfg, 'dev', use_augmentation=False)

    print('[DATASET CHECK] train samples:', len(ds_train))
    print('[DATASET CHECK] dev samples:', len(ds_dev))
    print('[DATASET CHECK] data_root used:', ds_train.data_root)

    # Print few labels
    for i in range(min(3, len(ds_train))):
        _, label = ds_train[i]
        print(f'sample {i} label_idx={int(label)}')


if __name__ == '__main__':
    main()
