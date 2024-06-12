# Open LLMs for Sequential Recommendation: SBBD 2024 Experiments

## Overview
This repository contains the scripts and resources for the experiments presented in the paper "An Analysis of Open Large Language Models in Sequential Recommendation Tasks" at SBBD 2024. The study investigates the effectiveness of open large language models (LLMs) in predicting the next item in a sequence with and without fine-tuning.

## Table of Contents
- [Introduction](#introduction)
- [Setup and Installation](#setup-and-installation)
- [Data](#data)
- [Experiments](#experiments)
  - [Zero-Shot Recommendations](#zero-shot-recommendations)
  - [Fine-Tuning](#fine-tuning)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This repository hosts the scripts and resources for the experiments conducted as part of the research paper "An Analysis of Open Large Language Models in Sequential Recommendation Tasks," presented at the SBBD 2024 conference. The study explores the potential of open large language models (LLMs) in enhancing recommendation systems by predicting the next item a user might be interested in, based on their previous interactions.

With the rise of generative AI, LLMs have gained significant attention for their versatility and capability to handle a variety of tasks beyond natural language processing. In this research, we specifically focus on the application of open LLMs, which are accessible and can be fine-tuned, making them particularly valuable for academic and practical applications.

Our experiments aim to assess the performance of these models in a sequential recommendation setting, comparing their effectiveness both with and without fine-tuning. By leveraging methods from existing literature, we demonstrate that open LLMs, even with fewer parameters, can outperform their closed counterparts, highlighting the importance of model tuning and data quality.

This repository includes all the necessary scripts to replicate our experiments, from setting up the environment and preparing the data to running the recommendation models and analyzing the results. We hope that our work contributes to the ongoing discourse on the application of LLMs in recommendation systems and serves as a useful resource for researchers and practitioners alike.

## Setup and Installation
Instructions on how to set up the environment and install the necessary dependencies. This could include:
- Prerequisites
- Installation steps
- Setting up virtual environments

Sure! Here's a text for the "Data" section in English:

## Data

The dataset utilized was MovieLens 100k dataset, a widely-used benchmark in the field of recommendation systems. The dataset consists of 100,000 ratings (from 1 to 5) provided by 943 users on 1,682 movies. Each user has rated at least 20 movies, making it a robust dataset for evaluating recommendation algorithms.

### Original Dataset
The MovieLens 100k dataset can be accessed and downloaded from the [GroupLens website](https://grouplens.org/datasets/movielens/100k/). It is available in a well-structured format, with separate files for user ratings, movie information, and user details. The dataset includes:

- **u.data**: Contains the user-item ratings.
- **u.item**: Provides metadata about the movies.
- **u.user**: Contains demographic information about the users.

### Pre-processed Dataset
For our experiments, we used a pre-processed version of the dataset provided by the authors of the baseline study. This pre-processed version involves filtering and structuring the data to focus on the most relevant user interactions and items, thus optimizing the dataset for sequential recommendation tasks.

The pre-processing steps included:
1. **Filtering Users**: Only users who have rated a least 20 moviess were retained.
2. **Candidate Items**: Generating a candidate set of items for each user based on their interaction history, following the methodology described in the baseline study.

The pre-processed dataset ensures that the experimental setup is aligned with the baseline study, allowing for a direct comparison of results. Detailed instructions for obtaining and preparing this pre-processed data are included in the repository, ensuring that you can replicate the experimental conditions accurately.

### Data Preparation
To prepare the data for use with our scripts, follow these steps:
1. Download the original MovieLens 100k dataset from the [GroupLens website](https://grouplens.org/datasets/movielens/100k/).
2. Follow the provided instructions in the `data_preparation.md` file to pre-process the data according to the methodology outlined above.
3. Ensure that the data files are placed in the appropriate directories as specified in the scripts.

By following these steps, you can ensure that your dataset is correctly prepared for the experiments, enabling accurate and reproducible results.

## Experiments

The experiments were designed to assess the models' capabilities in zero-shot settings and after fine-tuning. Below, we describe the algorithms used, the open LLMs evaluated, and the specifics of each experimental setup.

### Algorithms

We employed the ZeroShotNIR algorithm, which utilizes prompt-based techniques for generating recommendations without requiring any pre-existing examples (zero-shot). The algorithm involves three main steps:
1. Summarizing user preferences based on their interaction history.
2. Identifying representative items that best capture the user's preferences.
3. Recommending new items similar to the identified representative items from a candidate set.

#### Collaborative Filtering Configurations

To build the candidate set for recommendations, we utilized two collaborative filtering configurations:
1. **Original Configuration**: This configuration follows the setup used in the Wang & Lim study. It involves selecting the top M most similar users to the target user based on their interaction history. From these similar users, we identify the top S most popular items. Specifically, M=12 similar users and S=19 popular items were chosen.
   
2. **Expanded Configuration**: To explore the impact of a larger candidate set, we expanded the parameters. In this configuration, M=14 most similar users and S=38 popular items were selected. This expanded candidate set aimed to increase the diversity of potential recommendations, albeit at the cost of potentially diluting the recommendation accuracy due to the larger set of options.

The candidate set generation process involves the following steps:
1. **Vector Representation**: Convert the user's interaction history into a multi-hot vector, where each position represents an item, and a value of 1 indicates interaction with that item.
2. **Similarity Calculation**: Calculate the cosine similarity between the target user's vector and the vectors of all other users to find the most similar users.
3. **Candidate Set Formation**: Select the top M similar users and identify the top S most popular items among these users to form the candidate set.

### Open LLMs

The following open LLMs were used in our experiments:
- **Falcon**: Developed by the Technology Innovation Institute, available in various sizes, including 7B, 40B, and 180B parameters. It features rotary positional embeddings and hyperparameter optimizations.
- **Llama**: Created by Meta, available in sizes like 7B, 8B, 13B, and 70B parameters, with extended context windows and efficient tokenization.
- **Mistral**: Released by Mistral AI, available in configurations such as 7B, 8x7B, and 8x22B parameters, with sliding window attention and sparse mixture of experts.

### Zero-Shot Recommendations

The zero-shot recommendation experiment aimed to evaluate the models' ability to make accurate recommendations without any prior fine-tuning. We used the MovieLens 100k dataset, with prompts crafted to extract user preferences and generate item recommendations directly from the models. This setup tests the inherent capabilities of the LLMs to understand and predict user behavior based on limited input.

### Fine-Tuning: 80-20 Split

In the 80-20 fine-tuning experiment, we divided the dataset into 80% for training and 20% for testing. The goal was to improve the models' recommendation accuracy by allowing them to learn from a portion of the data. Fine-tuning involved adjusting the models using the LoRA technique to enhance their performance in predicting the next item in the sequence.

### Fine-Tuning: Cross Validation 4-Fold

For the cross-validation experiment, we used a 4-fold cross-validation approach to provide a robust evaluation of the models' performance. The dataset was split into four equal parts, with each part serving as a test set while the remaining three parts were used for training. This process was repeated four times, ensuring that each subset was used for validation exactly once. This method helps in assessing the generalizability and stability of the models' performance across different subsets of data.

### Hyperparameters

To ensure consistency and replicability across all experiments, we utilized the following hyperparameters for fine-tuning the models:

- **Temperature**: 0.1 – This parameter controls the randomness of the model's predictions. A lower temperature results in more conservative and focused predictions.
- **Penalty**: 1.15 – This parameter is used to avoid frequent repetitions in the generated responses.
- **Epochs**: 3 – The number of training epochs was set to 3, given the size of the dataset, to balance training time and model performance.
- **Learning Rate**: 2e-5 – The learning rate was selected to provide a balance between convergence speed and stability.

These hyperparameters were chosen to provide a good balance between model performance and training efficiency.

Through these experiments, we aim to provide comprehensive insights into the effectiveness of open LLMs in sequential recommendation tasks, highlighting the potential improvements achievable through fine-tuning and the inherent strengths of different models in zero-shot scenarios.

## Results
Summary of the results obtained from the experiments. Include:
- Comparison of different models
- Performance metrics used (e.g., HR@10, NDCG@10)
- Analysis of the findings

## Usage
Guidelines on how to use the scripts provided in the repository. This should cover:
- Running the experiments
- Evaluating new models
- Interpreting the results

## Contributing
Information on how others can contribute to the project. This could include:
- Contribution guidelines
- Code of conduct

## License
Details about the license under which the project is distributed. Typically, this section includes a link to the full license text.

## Contact
Contact information for the authors or maintainers of the repository. This could include:
- Email addresses
- Links to personal or professional profiles
