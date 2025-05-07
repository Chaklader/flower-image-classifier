import os
import shutil
from pathlib import Path

# Setup paths
base_dir = Path("Cat_Dog_data")
source_dir = base_dir / "PetImages"
train_dir = base_dir / "train"
test_dir = base_dir / "test"

# Create directories
(train_dir / "cats").mkdir(parents=True, exist_ok=True)
(train_dir / "dogs").mkdir(parents=True, exist_ok=True)
(test_dir / "cats").mkdir(parents=True, exist_ok=True)
(test_dir / "dogs").mkdir(parents=True, exist_ok=True)

# Function to move first N files
def move_files(source, dest, n, ext=".jpg"):
    files = list(source.glob(f"*{ext}"))
    for f in files[:n]:
        shutil.move(str(f), str(dest / f.name))

# Move training images (2500 each)
move_files(source_dir / "Cat", train_dir / "cats", 2500)
move_files(source_dir / "Dog", train_dir / "dogs", 2500)

# Move test images (500 each)
move_files(source_dir / "Cat", test_dir / "cats", 500)
move_files(source_dir / "Dog", test_dir / "dogs", 500)
