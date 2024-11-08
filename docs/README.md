# CoMix Documentation

## Overview
CoMix is a comprehensive toolkit for comic book analysis, providing tools for:
- Dataset unification and preprocessing
- Common comics objects detection (panels, characters, face, texts)
- Captioning of panels


## Project Structure
```
CoMix/
├── benchmarks/                     # Model implementations
│
│   ├── captioning/                 # Captioning models
│   │   ├── generate_captions.py    # Caption generation
│   │   ├── postprocessing.py       # Postprocessing
│   │   └── prompts.py              # Prompts for captioning
│
│   ├── detection/                  # Detection models
│   │   ├── dass.py                 # Yolox model for face/character detection
│   │   ├── faster_rcnn.py          # FasterRCNN model custom fine-tuned
│   │   ├── groundingdino.py        # GroundingDINO detection (zero-shot)
│   │   ├── magi.py                 # Magi character/panel/text detection
│   │   └── yolov8.py               # YOLOv8 model custom fine-tuned
│   │
│   └── weights/                    # Model weights
│
├── comix/
│   ├── evaluators/                 # Evaluation modules
│   │   ├── detection.py            # Detection metrics
│   │   ├── captioning.py           # Captioning metrics
│   │   └── detection_batch.py      # Batch evaluation
│   │
│   ├── process/                    # Task-specific modules
│   │   ├── comics.py               # Comics100 dataset processor
│   │   ├── dcm.py                  # DCM dataset processor
│   │   ├── ebdtheque.py            # eBDtheque dataset processor
│   │   ├── manga109.py             # Manga109 dataset processor
│   │   └── popmanga.py             # PopManga dataset processor
│   │
│   └── utils/                      # Utility functions
│
├── data/                           # Dataset folders
│
└── docs/                           # Documentation
```


## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (for GPU support)
- Git LFS (for model weights)

1. Clone the repository:
```bash
git clone https://github.com/emanueleviv/CoMix.git
cd CoMix
```

2. Install dependencies:
```bash
pip install -e .
```

3. Download model weights:
```bash  
# Create weights directory
mkdir -p benchmarks/weights
cd benchmarks/weights
# Download from provided links
# See specific model documentation for details
```

4. Ensure your data follows the required structure (see [Input Data Format](#input-data-format))

## Tasks

### 1. Detection
Panel, character, text, and face detection using various models:
- [Detection Documentation](tasks/detection/README.md)
- [Model Details](tasks/detection/generation.md)
- [Evaluation Metrics](tasks/detection/evaluation.md)

### 2. Captioning
Dense captioning of comic panels:
- [Captioning Documentation](tasks/captioning/README.md)
- [Generation Pipeline](tasks/captioning/generation.md)
- [Post-processing](tasks/captioning/postprocessing.md)
- [Evaluation](tasks/captioning/evaluation.md)

## Model Zoo
Pre-trained models are available for:
1. **Detection**
   - Faster R-CNN
   - DASS
   - YOLOv8
   - Grounding DINO
   - MAGI

2. **Captioning**
   - MiniCPM-V
   - Qwen2
   - Florence2
   - IDEFICS

Download links and setup instructions are provided in respective task documentation.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License.

## Citation
If you use this toolkit in your research, please cite:
```bibtex
@inproceedings{Vivoli2024ComicsDF,
  title={Comics Datasets Framework: Mix of Comics datasets for detection benchmarking},
  author={Emanuele Vivoli and Irene Campaioli and Mariateresa Nardoni and Niccol{\'o} Biondi and Marco Bertini and Dimosthenis Karatzas},
  booktitle={IEEE International Conference on Document Analysis and Recognition},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:271039307}
}

@article{Vivoli2024CoMixAC,
  title={CoMix: A Comprehensive Benchmark for Multi-Task Comic Understanding},
  author={Emanuele Vivoli and Marco Bertini and Dimosthenis Karatzas},
  journal={ArXiv},
  year={2024},
  volume={abs/2407.03550},
  url={https://api.semanticscholar.org/CorpusID:271038747}
}

@article{Vivoli2024ComiCapAV,
  title={ComiCap: A VLMs pipeline for dense captioning of Comic Panels},
  author={Emanuele Vivoli and Niccol{\'o} Biondi and Marco Bertini and Dimosthenis Karatzas},
  journal={ArXiv},
  year={2024},
  volume={abs/2409.16159},
  url={https://api.semanticscholar.org/CorpusID:272831696}
}
```

and the Survey on Comics Understanding:
```bibtex
@article{Vivoli2024OneMP,
  title={One missing piece in Vision and Language: A Survey on Comics Understanding},
  author={Emanuele Vivoli and Andrey Barsky and Mohamed Ali Souibgui and Artemis LLabres and Marco Bertini and Dimosthenis Karatzas},
  journal={ArXiv},
  year={2024},
  volume={abs/2409.09502},
  url={https://api.semanticscholar.org/CorpusID:272689435}
}
```