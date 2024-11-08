# fill this

before the postprocessing, the data should be in the following format:
```
data/predicts.caps/
├── [model-name]-cap/
│   ├── compiled_panels_annotations.csv
│   ├── gt_captions.csv
│   ├── gt_lists.csv
│   ├── [model-name]_captions.csv
│   └── [model-name]_lists.csv
```

after the postprocessing, the data should be in the following format:
```
data/predicts.caps/
├── eval/
│   ├── compiled_panels_annotations.csv
│   ├── gt_captions.csv
│   ├── gt_lists.csv
│   ├── [model-name]_captions.csv
│   └── [model-name]_lists.csv
```