# Anomaly Detection Framework with Contrastive Learning and GAN

This repository contains an implementation of an anomaly detection framework for multivariate time series data using PyTorch. The framework integrates contrastive learning and a Generative Adversarial Network (GAN) to address overfitting and learn robust representations.

## Features

- Data preprocessing with `MinMaxScaler` from scikit-learn
- Data augmentation using geometric distribution masks
- Transformer-based Autoencoder for time series reconstruction
- Contrastive loss for learning robust representations
- GAN architecture with a generator and discriminator
- Adversarial loss for training the GAN
- Anomaly detection based on reconstruction error

## Implementation

The code is organized into several components:

1. **Data Preprocessing**: The `preprocess_data` function normalizes the input data using `MinMaxScaler`.

2. **Data Augmentation**: The `geometric_mask` function applies a geometric distribution mask to augment the input time series data.

3. **Transformer-based Autoencoder**: The `TransformerAutoencoder` class defines the Autoencoder model, which uses PyTorch's `nn.TransformerEncoder` and `nn.TransformerDecoder` modules.

4. **Contrastive Loss**: The `contrastive_loss` function computes the contrastive loss between the input and reconstructed time series.

5. **GAN Models**: The `Generator` and `Discriminator` classes define the generator and discriminator models for the GAN architecture.

6. **Adversarial Loss**: The `adversarial_loss` function computes the adversarial loss for training the GAN.

7. **Training**: The `train` function orchestrates the training process, iterating over the data in batches. It performs data augmentation, trains the Autoencoder with the contrastive loss, trains the GAN with the adversarial loss, and updates the discriminator with samples from the Autoencoder's latent space.

8. **Anomaly Detection**: The `anomaly_detection` function takes the trained Autoencoder and the input data, reconstructs the data using the Autoencoder, computes the reconstruction error, and identifies anomalies based on a threshold.

## Usage

1. Prepare your multivariate time series dataset.
2. Import the necessary libraries and functions from the provided code.
3. Load your dataset and preprocess it using `preprocess_data`.
4. Set the desired hyperparameters (e.g., batch size, latent size, number of layers, and number of heads).
5. Call the `train` function with your data and hyperparameters to train the Autoencoder, Generator, and Discriminator models.
6. Use the `anomaly_detection` function with the trained Autoencoder and input data to identify anomalies based on the reconstruction error.

Note: You may need to adjust the model architectures and hyperparameters based on your specific dataset and requirements.

## Dependencies

- PyTorch
- scikit-learn

Make sure to install the required dependencies before running the code.
