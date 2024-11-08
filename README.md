# CoMix: Comics Dataset Framework for Comics Understanding

[![Package Version](https://img.shields.io/badge/version-1.0.0-yellow.svg)](https://github.com/emanuelevivoli/cdf)

The repo is under development. The codebase is called `comix`.
Please contribute to the project by reporting issues, suggesting improvements or adding new datasets.
We are currently working on refactoring the code to support:
- [ ] Automatic annotation (needs refactoring)
- [ ] Scraping new datasets (needs refactoring)
- [ ] Multi-task CoMix benchmarks (doing)


## Introduction
The purpose of this project is to replicate (on the validation set) the benchmarks presented in:
- [x] (detection) [Comics Datasets Framework: Mix of Comics datasets for detection benchmarking](https://arxiv.org/abs/2407.03540)
- [ ] (multitask) [CoMix: A Comprehensive Benchmark for Multi-Task Comic Understanding](https://arxiv.org/abs/2407.03550)
- [x] (captioning) [ComiCap: A VLMs pipeline for dense captioning of Comic Panels](https://arxiv.org/abs/2409.16159)

In particular, one of the main limitation when working in Comics/Manga datasets is the impossibility to share images.
To overcome this problem, we have created this framework that allows to use our (validation) annotations, and download the images from the original sources, without breaking the licenses.

The `comix` is using the following datasets:
- [x] DCM
- [x] comics
- [x] eBDtheque
- [x] PopManga
- [ ] Manga109

## Installation
The project is written in Python 3.8. To create a conda environment, consider using:
```bash
conda create --name myenv python=3.8
conda activate myenv
```

and to install the dependencies, run the following command:
```bash
pip install -e .
```
The above command will install the package `comix` in editable mode, so that you can modify the code and see the changes immediately. In the case of benchmarking the detection and captioning models, we will create separate conda environments to not conflict with the dependencies.

## Procedures
In general, this project is divided into the following steps:
- [x] Manually obtain and locate images and annotations in the right folder (e.g. `data/`)
- [x] Processing images to a unified naming and folder structure - `comix/process`
- [x] Model performances (use pre-trained or custom models on the data) - `benchmarks`
- [x] Evaluate model performances against provided Ground Truth - `comix/evaluators`

## Model performances and evaluation
In the `benchmarks` folder, we have multiple scripts to benchmark the models on the datasets, on various tasks.
The detection scripts produce a COCO-format json file which can be used by the `comix/evaluators/detection.py` script to evaluate the performances of the models.
The captioning scripts produce multiple `.txt` files, which can be postprocess to obtain a `captions.csv` and `objects.csv` files, used by the `comix/evaluators/captioning.py` script to evaluate the performances of the models.

## Documentation

The documentation is available in the `/docs` folder.

In particular:
```
docs/
├── README.md                   # Project overview, installation, quick start
├── datasets/                   # Dataset documentation
│   └── README.md               # Unified dataset info
└── tasks/                      # Task-specific documentation
    ├── detection.md            # Detection task
    │   ├── README.md           # Overview of detection pipeline
    │   ├── generation.md       # Detection models
    │   └── evaluation.md       # Metrics and evaluation
    └── captioning/             # Captioning task
        ├── README.md           # Overview of captioning pipeline
        ├── generation.md       # VLM caption generation details
        ├── postprocessing.md   # LLaMA post-processing
        └── evaluation.md       # Metrics and evaluation
```

Here are the most important documents:
- [docs/README.md](docs/README.md)
    - [docs/datasets/README.md](docs/datasets/README.md)
    - [docs/tasks/detection/README.md](docs/tasks/detection/README.md)
        - [docs/tasks/detection/generation.md](docs/tasks/detection/generation.md)
        - [docs/tasks/detection/evaluation.md](docs/tasks/detection/evaluation.md)
    - [docs/tasks/captioning/README.md](docs/tasks/captioning/README.md)
        - [docs/tasks/captioning/generation.md](docs/tasks/captioning/generation.md)
        - [docs/tasks/captioning/postprocessing.md](docs/tasks/captioning/postprocessing.md)
        - [docs/tasks/captioning/evaluation.md](docs/tasks/captioning/evaluation.md)
