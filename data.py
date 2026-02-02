import os
import shutil
import tarfile
import random
from pathlib import Path
import argparse

"""Helpers to extract provided archives and organize images into
dataset/{train,val,test}/{complete,defect}

Usage examples (see README for full instructions):
python data.py --archives tranistor.tar.xz screw.tar.xz --map tranistor:defect screw:complete
"""


def extract_archives(archives, extract_to='raw'):
    os.makedirs(extract_to, exist_ok=True)
    extracted_dirs = []
    for arc in archives:
        arc_path = Path(arc)
        if not arc_path.exists():
            print(f"Archive not found: {arc}")
            continue
        print(f"Extracting {arc} -> {extract_to}")
        try:
            with tarfile.open(arc_path) as tf:
                tf.extractall(path=extract_to)
                # many tarballs contain a top folder; note it
                # we won't attempt to automatically find it beyond listing
        except Exception as e:
            print(f"Failed to extract {arc}: {e}")
    # return list of immediate subdirectories
    for child in Path(extract_to).iterdir():
        if child.is_dir():
            extracted_dirs.append(str(child))
    return extracted_dirs


def _copy_images_to_split(src_dir, dest_root, class_label, splits=(0.7, 0.15, 0.15), seed=1234):
    files = [p for p in Path(src_dir).rglob('*') if p.is_file()]
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    sets = {
        'train': files[:n_train],
        'val': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:]
    }
    for split, items in sets.items():
        out_dir = Path(dest_root) / split / class_label
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in items:
            try:
                dest = out_dir / src.name
                shutil.copy2(src, dest)
            except Exception as e:
                print(f"Failed to copy {src} -> {dest}: {e}")


def prepare_dataset_from_extracted(extracted_dirs_map, dest_root='dataset', splits=(0.7, 0.15, 0.15), seed=1234):
    """
    extracted_dirs_map: dict mapping source dir -> target class label (complete|defect)
    Example: {'raw/tranistor':'defect', 'raw/screw':'complete'}
    """
    for src, label in extracted_dirs_map.items():
        if not Path(src).exists():
            print(f"Source folder does not exist: {src}")
            continue
        print(f"Organizing {src} -> class={label}")
        _copy_images_to_split(src, dest_root, label, splits=splits, seed=seed)
    print("Dataset prepared under:")
    print(Path(dest_root).resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archives', nargs='+', help='Paths to .tar.xz or .tar archives to extract')
    parser.add_argument('--map', nargs='+', help='Map archive base-name (or extracted folder name) to class label, e.g. tranistor:defect')
    parser.add_argument('--raw', default='raw', help='Raw extraction folder')
    parser.add_argument('--dest', default='dataset', help='Destination dataset folder')
    args = parser.parse_args()

    if args.archives:
        extract_archives(args.archives, extract_to=args.raw)

    mapping = {}
    if args.map:
        for m in args.map:
            if ':' not in m:
                print(f"Invalid map entry: {m}")
                continue
            src_name, label = m.split(':', 1)
            # look for a folder in raw that contains src_name
            found = None
            for child in Path(args.raw).iterdir():
                if src_name in child.name:
                    found = str(child)
                    break
            if not found:
                # allow absolute path
                if Path(src_name).exists():
                    found = src_name
                else:
                    print(f"Could not find extracted folder for {src_name} in {args.raw}")
                    continue
            mapping[found] = label

    if mapping:
        prepare_dataset_from_extracted(mapping, dest_root=args.dest)
    else:
        print("No mapping provided. Please create dataset manually under dataset/train, dataset/val, dataset/test with class subfolders 'complete' and 'defect'.")


if __name__ == '__main__':
    main()
