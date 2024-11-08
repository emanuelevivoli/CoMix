# Fast-track Datasets
In general, you should be able to download the datasets from the sources. However, we provide a fast-track for the datasets that are available in the CoMix repository (only validation) so that you can start **very quick**.

**Note:**
Contact us if you need access to the fast-track datasets and we'll provide you with a link.
If you get access, you just need to unzip it and locate it under the `data/datasets.unify` folder (just rename `datasets.unify.fast` to `datasets.unify`).

## Image Structure

The datasets should be located in the `data/datasets` folder, following the structure:
```
data/datasets.unify/                            # rename datasets.unify.fast to datasets.unify
    ├── compiled_panels_annotations.csv         # panel detection annotations
    ├── [subdb]/
    │   ├── [comic_no]/
    │   │   └── [page_no].jpg
    ...
data/predicts.caps/                            # content of cap-anns-val
    ├── gt_captions.csv
    ├── gt_lists.csv
    ├── minicpm_captions.csv
    └── minicpm_lists.csv
```

For the captioning task, the images are the same, but we also provide the `panel detection` (manual annotations), called `compiled_panels_annotations.csv`, that you should locate under the `data/datasets.unify` folder.

## Annotations Structure

The (validation) annotations are available in the [open access Drive folder](https://drive.google.com/drive/folders/1i4c3ZXBEjGPAkQd2coS0_Ir2wz2q98oo?usp=sharing). There you can find:
- the `weights` folder needed for the detection benchmarks;
- the `det-anns-val` validation ground truth annotation in `.json` (coco) and `.xml` (custom format) for detection;
- the `splits` folder with the `.csv` files for the splits.
- the `cap-anns-val` validation ground truth _captions_ and _object-list_ annotation in `.csv`, for the GT and MiniCPM predictions. This should be located under `data/predicts.caps`.

## Evaluation

To evaluate the predictions, you can use the scripts in `comix/evaluators` folder.
If you want to evaluate your predictions, first generate the `.json` files (coco format) for the detection and the `.csv` files for the captions and objects-lists, and then run the evaluation scripts.
