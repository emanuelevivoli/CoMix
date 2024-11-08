from setuptools import setup, find_packages

setup(
    name='comix',
    version='0.2.0',
    author='Emanuele Vivoli',
    author_email='emanuele.vivoli@gmail.com',
    description='A pip package to convert and manage Comic Datasets to Unified XML format.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/emanuelevivoli/comix',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'pillow',
        'lxml',
        'bs4',
        'beautifulsoup4',
        'matplotlib',
        'tqdm',
        'requests',
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'torch',
        'torchaudio',
        'torchvision',
        'transformers',
        'tokenizers',
        'manga-py',
        'mangadex',
        'mloader',
        'boto3',
        'pycocotools',
        'ultralytics',
        'einops',
        'shapely',
        'timm',
        'termcolor',
        'rarfile',
        'wandb',
        # Add any other dependencies your package needs here.
        # These will be installed when your package is installed.
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Update the license as appropriate
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    # Add any additional keywords relevant to your package
    keywords='comics, dataset, xml, conversion, dcm, manga, mangadex, mloader',
)
