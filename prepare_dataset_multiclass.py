"""
prepare_dataset_multiclass.py

Creates multi-class dataset with specific defect types.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
ALLOWED_EXT = {'.png', '.jpg', '.jpeg'}
SPLITS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

# Define all classes
CLASSES = [
    'complete',
    'manipulated_front',
    'scratch_head',
    'scratch_neck',
    'thread_side',
    'thread_top',
    'bent_lead',
    'cut_lead',
    'damaged_case',
    'misplaced'
]

def ensure_dirs():
    """Create output directory structure for all classes."""
    dataset_root = Path('dataset_multiclass')
    for split in SPLITS.keys():
        for cls in CLASSES:
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
    """Collect images for each class."""
    images = defaultdict(list)
    
    # Collect 'complete' from screw good images
    screw_train_good = Path('raw/screw/screw/train/good')
    screw_test_good = Path('raw/screw/screw/test/good')
    
    for path in [screw_train_good, screw_test_good]:
        if path.exists():
            files = get_image_files(path)
            images['complete'].extend(files)
            print(f"Found {len(files)} complete images from {path}")
    
    # Collect 'complete' from transistor good images
    transistor_train_good = Path('raw/transistor/transistor/train/good')
    transistor_test_good = Path('raw/transistor/transistor/test/good')
    
    for path in [transistor_train_good, transistor_test_good]:
        if path.exists():
            files = get_image_files(path)
            images['complete'].extend(files)
            print(f"Found {len(files)} complete images from {path}")
    
    # Collect screw defects
    screw_defects = {
        'manipulated_front': 'raw/screw/screw/test/manipulated_front',
        'scratch_head': 'raw/screw/screw/test/scratch_head',
        'scratch_neck': 'raw/screw/screw/test/scratch_neck',
        'thread_side': 'raw/screw/screw/test/thread_side',
        'thread_top': 'raw/screw/screw/test/thread_top'
    }
    
    for class_name, path_str in screw_defects.items():
        path = Path(path_str)
        if path.exists():
            files = get_image_files(path)
            images[class_name].extend(files)
            print(f"Found {len(files)} {class_name} images")
    
    # Collect transistor defects
    transistor_defects = {
        'bent_lead': 'raw/transistor/transistor/test/bent_lead',
        'cut_lead': 'raw/transistor/transistor/test/cut_lead',
        'damaged_case': 'raw/transistor/transistor/test/damaged_case',
        'misplaced': 'raw/transistor/transistor/test/misplaced'
    }
    
    for class_name, path_str in transistor_defects.items():
        path = Path(path_str)
        if path.exists():
            files = get_image_files(path)
            images[class_name].extend(files)
            print(f"Found {len(files)} {class_name} images")
    
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
            dst_dir = Path('dataset_multiclass') / split_name / class_label
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src_file in files:
                copy_with_dedup(src_file, dst_dir)
                counts[split_name][class_label] += 1
    
    return counts

def print_summary(counts):
    """Print dataset summary."""
    print("\n" + "="*70)
    print("Multi-Class Dataset Summary")
    print("="*70)
    for split in SPLITS.keys():
        print(f"\n{split.upper()}/")
        total = 0
        for cls in CLASSES:
            count = counts[split][cls]
            if count > 0:
                print(f"  {cls:20s}: {count:4d}")
                total += count
        print(f"  {'TOTAL':20s}: {total:4d}")

if __name__ == '__main__':
    print("Preparing multi-class dataset...")
    ensure_dirs()
    images = collect_class_images()
    counts = split_and_copy(images)
    print_summary(counts)
    print("\nMulti-class dataset ready at: dataset_multiclass/")
