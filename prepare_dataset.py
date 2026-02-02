"""
prepare_dataset.py

Converts extracted MVTec folders (screw, transistor) into Phase-1 dataset format.
- Copies screw/train/good/ -> complete class
- Copies transistor/test/* (except good) -> missing class
- Splits into train/val/test (80/10/10)
- Avoids overwriting with counter suffix
- Prints final counts per folder
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
ALLOWED_EXT = {'.png', '.jpg', '.jpeg'}
SPLITS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

def ensure_dirs():
    """Create output directory structure."""
    dataset_root = Path('dataset')
    for split in SPLITS.keys():
        for cls in ['complete', 'missing']:
            (dataset_root / split / cls).mkdir(parents=True, exist_ok=True)

def get_image_files(src_dir):
    """Recursively find all image files."""
    src_path = Path(src_dir)
    if not src_path.exists():
        print(f"Warning: {src_dir} does not exist")
        return []
    files = [p for p in src_path.rglob('*') if p.is_file() and p.suffix.lower() in ALLOWED_EXT]
    return files

def copy_with_dedup(src_file, dst_dir):
    """Copy file to dst_dir, adding counter if name exists."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    dst_file = dst_dir / src_file.name
    if not dst_file.exists():
        shutil.copy2(src_file, dst_file)
        return True
    
    # Add counter suffix
    stem = src_file.stem
    suffix = src_file.suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        dst_file = dst_dir / new_name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
            return True
        counter += 1

def collect_class_images():
    """Collect images for 'complete' and 'missing' classes."""
    images = defaultdict(list)
    
    # Collect 'complete' from screw/train/good
    screw_good = Path('raw/screw/screw/train/good')
    if screw_good.exists():
        files = get_image_files(screw_good)
        images['complete'].extend(files)
        print(f"Found {len(files)} complete (good) images from screw/train/good")
    
    # Collect 'complete' from screw/test/good
    screw_test_good = Path('raw/screw/screw/test/good')
    if screw_test_good.exists():
        files = get_image_files(screw_test_good)
        images['complete'].extend(files)
        print(f"Found {len(files)} complete (good) images from screw/test/good")
    
    # Collect 'complete' from transistor/train/good
    transistor_good = Path('raw/transistor/transistor/train/good')
    if transistor_good.exists():
        files = get_image_files(transistor_good)
        images['complete'].extend(files)
        print(f"Found {len(files)} complete (good) images from transistor/train/good")
    
    # Collect 'complete' from transistor/test/good
    transistor_test_good = Path('raw/transistor/transistor/test/good')
    if transistor_test_good.exists():
        files = get_image_files(transistor_test_good)
        images['complete'].extend(files)
        print(f"Found {len(files)} complete (good) images from transistor/test/good")
    
    # Collect 'missing' from screw/test/* except 'good'
    screw_test = Path('raw/screw/screw/test')
    if screw_test.exists():
        for subdir in screw_test.iterdir():
            if subdir.is_dir() and subdir.name.lower() != 'good':
                files = get_image_files(subdir)
                images['missing'].extend(files)
                print(f"Found {len(files)} missing/defect images from screw/{subdir.name}")
    
    # Collect 'missing' from transistor/test/* except 'good'
    transistor_test = Path('raw/transistor/transistor/test')
    if transistor_test.exists():
        for subdir in transistor_test.iterdir():
            if subdir.is_dir() and subdir.name.lower() != 'good':
                files = get_image_files(subdir)
                images['missing'].extend(files)
                print(f"Found {len(files)} missing images from transistor/{subdir.name}")
    
    return images

def split_and_copy(images_dict):
    """Split images into train/val/test and copy to dataset."""
    random.seed(SEED)
    counts = defaultdict(lambda: defaultdict(int))
    
    for class_label, image_list in images_dict.items():
        if not image_list:
            print(f"Warning: No images for class '{class_label}'")
            continue
        
        random.shuffle(image_list)
        n = len(image_list)
        n_train = int(n * SPLITS['train'])
        n_val = int(n * SPLITS['val'])
        
        splits = {
            'train': image_list[:n_train],
            'val': image_list[n_train:n_train + n_val],
            'test': image_list[n_train + n_val:]
        }
        
        for split_name, files in splits.items():
            dst_dir = Path('dataset') / split_name / class_label
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src_file in files:
                copy_with_dedup(src_file, dst_dir)
                counts[split_name][class_label] += 1
    
    return counts

def print_summary(counts):
    """Print dataset summary."""
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    for split in SPLITS.keys():
        print(f"\n{split.upper()}/")
        total = 0
        for cls in ['complete', 'missing']:
            count = counts[split][cls]
            print(f"  {cls:12s}: {count:4d}")
            total += count
        print(f"  {'total':12s}: {total:4d}")

if __name__ == '__main__':
    print("Preparing dataset...")
    ensure_dirs()
    images = collect_class_images()
    counts = split_and_copy(images)
    print_summary(counts)
    print("\nDataset ready at: dataset/")
