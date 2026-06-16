# Hand-Gesture Recognition (CNN)

A convolutional neural network that classifies static hand gestures from the
**LeapGestRecog** dataset, built with TensorFlow / Keras. The script covers the
full pipeline: data discovery, augmentation, training with callbacks, and
evaluation.

## Pipeline

1. **Data discovery** — walks the dataset directory (subjects → gesture folders),
   collects image paths and labels into a `pandas` DataFrame.
2. **Splitting** — train / validation / test split via `train_test_split`.
3. **Augmentation & loading** — `ImageDataGenerator` for rescaling and on-the-fly
   augmentation.
4. **Model** — a `Sequential` CNN: stacked `Conv2D` + `MaxPooling2D` blocks with
   `BatchNormalization` and `Dropout`, then `Dense` layers.
5. **Training** — with `EarlyStopping` and `ReduceLROnPlateau` callbacks.
6. **Evaluation** — accuracy/loss curves (matplotlib) plus a confusion matrix and
   `classification_report` (scikit-learn / seaborn).

## Requirements

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

## Usage

Download the [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
dataset and point the script at it by editing the `DATASET_PATH` constant near the
top of `main.py`, then:

```bash
python main.py
```

> **Note:** the dataset is not bundled with this repository, and `DATASET_PATH`
> must be set to a local copy before running.
