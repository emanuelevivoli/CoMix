import os
import argparse
import csv
import shutil
from comix.utils import generate_hash

def generate_file_mapping(root_path):
    file_mapping = {}
    subdir_path = os.path.join(root_path, 'images')
    for folder_name in os.listdir(subdir_path):
        folder_path = os.path.join(subdir_path, folder_name)
        if os.path.isdir(folder_path):
            sorted_files = sorted(os.listdir(folder_path))
            for idx, filename in enumerate(sorted_files, start=1):
                base_name, extension = os.path.splitext(filename)
                new_filename = f"{idx:03d}"
                old_path = os.path.join(folder_name, base_name)
                new_path = os.path.join(folder_name, new_filename)
                file_mapping[old_path] = new_path
                os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename + extension))
    return file_mapping

def parse_args_preprocess():
    parser = argparse.ArgumentParser(description='Rename files')
    parser.add_argument('--root', default="data/datasets/DCM", type=str, help='Root path to the DCM dataset')
    args = parser.parse_args()
    return args

# Parse Command Line Arguments
def parse_args_hash():
    parser = argparse.ArgumentParser(description='Process DCM dataset.')
    parser.add_argument('-i', '--input-path', type=str, help='Path to the DCM folder', default='data/datasets/DCM')
    parser.add_argument('-o', '--output-path', type=str, help='Path to the output folder',
                        default='data/datasets.unify/DCM')
    parser.add_argument('-l', '--limit', type=int, help='Limit the number of books processed')
    parser.add_argument('--ex-text', action='store_true', help='Include external text lines from the output')
    parser.add_argument('--override', action='store_true', help='Override existing image folders if they exist')
    args = parser.parse_args()
    return args


# Main Processing
def main(args):
    # Directories setup
    images_dir = os.path.join(args.output_path, 'images')

    os.makedirs(images_dir, exist_ok=True)

    input_images = os.path.join(args.input_path, 'images')

    # CSV for hash mapping
    book_hash_map = {}
    csv_path = os.path.join(args.output_path, 'book_hash_mapping.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['hash_code', 'book_name'])

        # Iterate over each book in the input directory
        for i, book_name in enumerate(os.listdir(input_images)):
            input_book_path = os.path.join(input_images, book_name)
            if os.path.isdir(input_book_path):
                # Generate hash code for book name
                hash_code = generate_hash(book_name)

                # Copy images
                output_images_dir = os.path.join(images_dir, hash_code)
                if not os.path.exists(output_images_dir) or args.override:
                    if os.path.exists(output_images_dir):
                        shutil.rmtree(output_images_dir)
                    shutil.copytree(input_book_path, output_images_dir)
                else:
                    print(f"Skipping existing directory: {output_images_dir}")

                # Write the mapping to CSV
                csv_writer.writerow([hash_code, book_name])
                book_hash_map[book_name] = hash_code

            # use this to limit the number of books processed
            if args.limit and i >= args.limit:
                break


# Main execution
if __name__ == "__main__":
    args = parse_args_preprocess()
    file_mapping = generate_file_mapping(args.root)
    print("Files renamed")
    args = parse_args_hash()
    main(args)