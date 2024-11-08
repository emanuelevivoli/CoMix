# Detection benchmarks

The `benchmarks/detection` folder contains the code for the following models:
- `Faster R-CNN`
- `DASS`
- `YOLOv8`
- `Grounding Dino`
- `Magi`
- `SSD` (to be added)

The benchmarks are run on the validation split of the `CoMix` dataset, which comprehends:
- Comics-style:
    - comics100 (val, test, dev[todo])
    - DCM (val, test)
    - eBDtheque (val, test)
- Manga-style:
    - PopManga (val, test)
    - Manga109[todo] (val, test, dev)

For these datasets, follow the instructions in the [CoMix/README.md](../../../README.md) and then [datasets/README.md](../../datasets/README.md) files to preprocess and convert the data.

Here you can find the instructions for running the models [generation](generation.md) and evaluating the predictions [evaluation](evaluation.md).
