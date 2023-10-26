# Frenchify-NeuroTranslate: Bi-Directional LSTM English to French Translation
This repository contains an implementation of a Bi-Directional Long Short-Term Memory (LSTM) model for translating English sentences to French. The project involves natural language processing (NLP), deep learning, and sequence-to-sequence modeling.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Example Predictions](#example-predictions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Machine translation is a challenging NLP task, and this project aims to provide a deep learning solution using Bi-Directional LSTMs. The model is trained on a dataset containing English and French sentence pairs. After training, the model can be used to translate English text into French.

## Requirements

This project is developed using Python and the following libraries:

- Pandas
- NumPy
- NLTK
- Gensim
- SpaCy
- Plotly
- Keras
- TensorFlow
- Scikit-Learn
- Seaborn
- Matplotlib
- WordCloud

You can install these libraries using `pip` and manage dependencies with a virtual environment.

```bash
pip install -r requirements.txt
```

## Data Preprocessing

The project includes data preprocessing steps such as removing punctuation, tokenization, and padding sequences. The provided dataset ('fra.txt') is cleaned and prepared for training.

## Model Architecture

The core of this project is a Bi-Directional LSTM model. The architecture includes embedding layers, Bidirectional LSTMs, Batch Normalization, and TimeDistributed Dense layers. The model is designed to perform sequence-to-sequence translation.

## Training

The model is trained using a subset of the dataset. The training process involves optimizing for categorical cross-entropy loss. Training details, including accuracy and loss over epochs, can be found in the `mo` variable.

## Parallelization during Model Training

In this project, we have leveraged the power of parallelization using TensorFlow's `CentralStorageStrategy`. This strategy allows for efficient distribution of computational tasks, resulting in faster training times and improved model performance. By parallelizing the training process, we have significantly enhanced the scalability and training efficiency of the Bi-Directional LSTM model.

The use of `CentralStorageStrategy` ensures that the available resources are utilized optimally, making this project suitable for large datasets and complex deep learning models.

## Evaluation

The model's performance is evaluated on a test dataset. The accuracy of the translation can be measured by comparing predicted translations to the actual translations.

## Usage

To use this model for translation, you can follow the code provided in the script. You can load the pre-trained model and make predictions on your own English text.

## Contributing

If you'd like to contribute to this project, feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
