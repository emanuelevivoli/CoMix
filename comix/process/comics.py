import os
import argparse
import csv
import shutil
from pathlib import Path
from comix.utils import generate_hash

def generate_file_mapping(root_path):
    file_mapping = {}
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for filename in files:
                base_name, extension = os.path.splitext(filename)
                new_filename = base_name.zfill(3)  # Zero-padding the base name
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename + extension)
                file_mapping[old_path] = new_path
                os.rename(old_path, new_path)
    return file_mapping

def parse_args_preprocess():
    parser = argparse.ArgumentParser(description='Rename files in comics dataset')
    parser.add_argument('--root', type=str, default='data/datasets/comics/books',
                        help='Root path to the comics dataset')
    args = parser.parse_args()
    return args

def parse_args_hash():
    parser = argparse.ArgumentParser(description='Process Comics dataset.')
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input dataset folder',
                        default='data/datasets/comics')
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output folder',
                        default='data/datasets.unify/comics')
    parser.add_argument('-l', '--limit', type=int, help='Limit the number of books processed', default=None)
    parser.add_argument('-ov', '--override', action='store_true', help='Override existing image folders if they exist')
    parser.add_argument('-b', '--books', default=None, help='Books to process', nargs='+')
    args = parser.parse_args()
    return args


# Main Processing Function
def process_comics_dataset(args):
    # Paths
    dataset_path = Path(args.input_path)
    data_path = dataset_path / 'books'

    books = args.books

    # Create Output Directories
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Create images and annotations folders
    images_path = Path(args.output_path) / 'images'
    if not os.path.exists(images_path):
        Path(images_path).mkdir(parents=True, exist_ok=True)

    mapping_csv_path = os.path.join(args.output_path, 'book_name_mapping.csv')

    with open(mapping_csv_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['hash_code', 'book_name'])

        # Process Books
        books_processed = 0
        for book_id in sorted(os.listdir(data_path)):
            if args.limit and books_processed >= args.limit:
                break

            book_hash = generate_hash(book_id)
            if books is not None and book_id not in books:
                continue

            # Copy images
            input_book_path = os.path.join(data_path, book_id)
            output_images_dir = os.path.join(images_path, book_hash)
            if not os.path.exists(output_images_dir) or args.override:
                if os.path.exists(output_images_dir):
                    shutil.rmtree(output_images_dir)
                shutil.copytree(input_book_path, output_images_dir)
            else:
                print(f"Skipping existing directory: {output_images_dir}")

            # Write the mapping to CSV
            csv_writer.writerow([book_hash, book_id])

            books_processed += 1


# PREPROCESS
args = parse_args_preprocess()
file_mapping = generate_file_mapping(args.root)
print("Files renamed.")
# PROCESS
args = parse_args_hash()
process_comics_dataset(args)