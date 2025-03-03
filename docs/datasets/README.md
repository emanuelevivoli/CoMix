# Datasets
The procedures to get and process the datasets are described here.

## Structure

This is the folder structure you need to handle data used in the CoMix repository. The data is stored in the following sub-folders:
- original datasets
    - `data/datasets`: contains the downloaded raw data (from other sources), used in the CoMix repository.
- processed datasets
    - `data/datasets.unify`: contains the unified data after having processed all the raw images and the splits (this folder is already present).
- unified datasets
    - `data/comix.coco`: in order to evaluate the models, the data is converted to COCO format.
- predictions
    - `predicts.coco`: predictions for detection models
    - `predicts.caps`: captioning

## Data Collection

### eBDtheque

The eBDtheque dataset can be downloaded from the [website](https://ebdtheque.univ-lr.fr/registration), after registration. Once you have downloaded the dataset, ONLY place the `Pages` folder into the `datasets/eBDtheque` folder.

#### Convert images

To convert the images of eBDtheque to the unified format, run the following command:
```bash
$ python comix/process/ebdtheque.py
```
Check the `comix/process/ebdtheque.py` file for the arguments, if you want to change the default values.

### Manga109
According to the [license of Manga109](http://www.manga109.org/en/download.html), the redistribution of the images of Manga109 is not permitted.
Thus, you should download the images of Manga109 via the [Manga109 webpage](http://www.manga109.org/en/download.html).

After downloading, unzip `Manga109.zip` into the folder `datasets`. Move all the contents of `Manga109_released_x` to the upper folder, then delete the empty directory `Manga109_released_x`.

Remove unused files:
```bash
cd data/datasets/Manga109
rm -rf annotations.v20*
rm -rf annotations
```

The folder structure should look like this:
```bash
datasets/
└── Manga109
    ├── images
    ├── books.txt
    ├── readme.txt
```

#### Convert images
To convert the images of Manga109 to the unified format, run the following command:
```bash
$ python comix/process/manga109.py
```
which has the following arguments:
- `--input-path`: path to the Manga109 folder (default: `data/datasets/Manga109`)
- `--output-path`: path to the output folder (default: `data/datasets.unify/Manga109`)
- `--override`: override the existing images, annotations are always overwritten (default: `False`)
- `--limit`: stop after the first `{limit}` books (default: `None`)

### DCM

After downloading the dataset from [here](https://gitlab.univ-lr.fr/crigau02/dcm-dataset), unzip `DCM_dataset_public_images.zip` into the folder `datasets`. Rename the extracted directory as `DCM` and delete the zip file.

#### Convert images
The DCM dataset needs to be preprocessed before being converted into the unified format. To preprocess the DCM dataset (jpg renaming) and then convert images to the unified format, run the following command:
```bash
$ python comix/process/dcm.py
```
In DCM original enumeration of images starts from '001' rather than '000'. We decided to keep it.

### Comics

Download the [original page images](https://obj.umiacs.umd.edu/comics/raw_page_images.tar.gz). Unzip `raw_pages_images.tar.gz` into the folder `datasets` and rename the extracted folder `books`. Then, move this folder in an upper new created directory named `comics`.

The folder hierarchy should look like this:
```bash
datasets/
└── comics
    ├── books
```

#### Convert images
The Comics dataset needs to be preprocessed before being converted into the unified format. To preprocess the Comics dataset (jpg renaming) and then convert images to the unified format, run the following command:
```bash
$ python comix/process/comics.py
```
In Comics dataset some images are not viewable (usually first/last ones). We renamed them anyway.

Check the `comix/process/comics.py` file for the arguments, if you want to change the default values.

### PopManga

To download the dataset, please refer to [magi repository](https://github.com/ragavsachdeva/magi). After downloading, please locate `Popmanga` folder into `data/datasets` and rename it into `popmanga`. Then, inside the folder delete `annotations`.

#### Convert images

Now, you can convert the images of comics to the unified format by running the following command:
```bash
$ python comix/process/popmanga.py
```

### SPLITS

In the path `data/datasets.unify/name_of_the_dataset/splits` the splits are available for every dataset except for `Manga109`. In particular `val.csv` and `test.csv` are available for every dataset. Furthermore, in `comics` there is also `train.csv`
