import os
import csv
import shutil
import argparse
from comix.utils import generate_hash

# Function to parse filename and extract information
def parse_filename(filename):
    # author _ (book_name) _ page_number . extension
    parts = filename.rsplit('_', 2)
    # author, (book_name) , page_number . extension
    return parts[0], '_'.join(parts[1:-1]), parts[-1].split('.')[0]

# Parse Command Line Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Process eBDtheque dataset.')
    parser.add_argument('-i', '--input-path', type=str, help='Path to the eBDtheque folder', default='data/datasets/eBDtheque')
    parser.add_argument('-o', '--output-path', type=str, help='Path to the output folder', default='data/datasets.unify/eBDtheque')
    parser.add_argument('-l', '--limit', type=int, help='Limit the number of books processed')
    parser.add_argument('--ex-text', action='store_true', help='Include external text lines from the output')
    parser.add_argument('--override', action='store_true', help='Override existing image folders if they exist')
    args = parser.parse_args()
    return args

# Main Processing
def main(args):
    # Processing logic (similar to the existing script but using new models and utilities)
    input_images_path = os.path.join(args.input_path, 'Pages')
    output_images_path = os.path.join(args.output_path, 'images')
    os.makedirs(output_images_path, exist_ok=True)

    # Read input SVG files, process them and write output XML files
    csv_path = os.path.join(args.output_path, 'book_hash_mapping.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['author', 'book_name', 'hash'])
        
        books = {}
        for img_file in sorted(os.listdir(input_images_path)):
            if img_file.endswith('.jpg'):
                author_name, book_name, page_number = parse_filename(img_file)
                book_id = f'{author_name}_{book_name}'
                if book_id not in books:
                    book_hash = generate_hash(book_id)
                    books[book_id] = {'pages': [], 
                                    'hash': book_hash, 
                                    'author': author_name, 
                                    'title': book_name}
                    csv_writer.writerow([author_name, book_name, book_hash])

                if args.limit and len(books) >= args.limit:
                    break

        # Write and Organize Output
        for book_id, book_data in books.items():
            book_hash = book_data['hash']

            book_images_dir = os.path.join(output_images_path, book_hash)
            if args.override and os.path.exists(book_images_dir):
                shutil.rmtree(book_images_dir)
            os.makedirs(book_images_dir, exist_ok=True)

            img_files = sorted(filter(lambda f: f.startswith(book_id), os.listdir(input_images_path)))
            # Copy images from the input to the output directory
            for page_counter, img_file in enumerate(img_files):
                new_img_filename = f"{str(page_counter).zfill(3)}.jpg"
                shutil.copy(os.path.join(input_images_path, img_file), os.path.join(book_images_dir, new_img_filename))

            print(f"Processed book: {book_hash}")

        print("Process completed.")


if __name__ == "__main__":
    args = parse_args()
    main(args)