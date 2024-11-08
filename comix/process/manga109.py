import os
import csv
import shutil
import argparse
from comix.utils import generate_hash

# Parse Command Line Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Process Manga109 dataset.')
    parser.add_argument('-i', '--input-path', type=str, help='Path to the Manga109 folder', default='data/datasets/Manga109')
    parser.add_argument('-o', '--output-path', type=str, help='Path to the output folder', default='data/datasets.unify/Manga109')
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

    # CSV file for mapping hash codes to book names
    mapping_csv_path = os.path.join(args.output_path, 'book_name_mapping.csv')
    with open(mapping_csv_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
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
            
            # use this to limit the number of books processed
            if args.limit is not None and i >= args.limit:
                break

if __name__ == "__main__":
    args = parse_args()
    main(args)