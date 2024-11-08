import hashlib

# Function to generate a hash
def generate_hash(name, length=8):
    return hashlib.md5(name.encode()).hexdigest()[:length]

# Generate unique image ID from book and page IDs
def get_image_id(book_id, page_id):
    book_id_int = int(book_id, 16)
    return book_id_int * 10000 + int(page_id)