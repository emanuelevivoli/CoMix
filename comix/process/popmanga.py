import os
import csv
import shutil
import argparse
from PIL import Image
from pathlib import Path
from comix.utils import generate_hash

def main(args):
    images_path = Path(args.input_path) / 'images'
    output_images_path = Path(args.output_path) / 'images'

    output_images_path.mkdir(parents=True, exist_ok=True)

    mapping_csv_path = Path(args.output_path) / 'book_chapter_hash_mapping.csv'

    book_hash_map = {}
    with open(mapping_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['book_chapter_hash', 'new_image_name', 'book_name', 'chapter_name', 'original_image_name'])

        for book_path in images_path.iterdir():
            if book_path.is_dir():
                book_name = book_path.name
                chapter_names = sorted(os.listdir(book_path))
                for chapter_name in chapter_names:
                    chapter_path = book_path / chapter_name
                    if not chapter_path.is_dir():
                        continue

                    book_chapter_hash = generate_hash(f'{book_name}_{chapter_name}')
                    output_chapter_images_path = output_images_path / book_chapter_hash
                    output_chapter_images_path.mkdir(exist_ok=True)

                    images_names = sorted([i for i in os.listdir(chapter_path) if i.endswith('.jpg')])
                    for image_no, image_file in enumerate(images_names):
                        new_image_name = f'{image_no:03d}'
                        try:
                            img = Image.open(os.path.join(chapter_path, image_file))
                            shape = img.size
                            if shape[0] == 0 or shape[1] == 0:
                                raise ValueError(f"Invalid image shape: {shape}")
                        except:
                            print(f"Error opening {chapter_path / image_file}")
                            continue
                        shutil.copy(chapter_path / image_file, output_chapter_images_path / f'{new_image_name}.jpg')

                        csv_writer.writerow([book_chapter_hash, new_image_name, book_name, chapter_name, image_file])
                        book_hash_map[f'{book_name}/{chapter_name}/{image_file}'] = (book_chapter_hash, new_image_name)

def parse_args():
    parser = argparse.ArgumentParser(description='Process PopManga dataset.')
    parser.add_argument('-i', '--input-path', type=str, help='Path to the PopManga folder', default='data/datasets/popmanga')
    parser.add_argument('-o', '--output-path', type=str, help='Path to the output folder', default='data/datasets.unify/popmanga')
    parser.add_argument('-l', '--limit', type=int, help='Limit the number of books processed')
    parser.add_argument('--ex-text', action='store_true', help='Include external text lines from the output')
    parser.add_argument('--override', action='store_true', help='Override existing image folders if they exist')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)