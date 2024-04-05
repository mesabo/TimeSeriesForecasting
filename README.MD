# @Author

Created on 20/03/2024
🚀 Welcome to the Awesome Python Script 🚀

- User: mesabo
- Email: mesabo18@gmail.com
- git: messouaboya17@gmail.com
- Github: https://github.com/mesabo
- Univ: Hosei University
- Dept: Science and Engineering
- Lab: Prof YU Keping's Lab

## Time Series Multivariate Multi-steps Forecasting Project

This project aims to provide a flexible and customizable solution for Time Series Multivariate Multi-steps forecasting.
It incorporates various models implemented in PyTorch 2.2.0, allowing users to choose from a wide range of models based
on their specific forecasting needs. The project supports forecasting for different time horizons, from short-term to
long-term, and provides implementations for the following models:

- GRU_MODEL
- CNN_MODEL
- BiLSTM_MODEL
- BiGRU_MODEL
- LSTM_ATTENTION_MODEL
- GRU_ATTENTION_MODEL
- CNN_ATTENTION_MODEL
- BiLSTM_ATTENTION_MODEL
- BiGRU_ATTENTION_MODEL
- CNN_LSTM_MODEL
- CNN_GRU_MODEL
- CNN_BiLSTM_MODEL
- CNN_BiGRU_MODEL
- CNN_LSTM_ATTENTION_MODEL
- CNN_GRU_ATTENTION_MODEL
- CNN_BiLSTM_ATTENTION_MODEL
- CNN_BiGRU_ATTENTION_MODEL
- CNN_ATTENTION_LSTM_ATTENTION_MODEL
- CNN_ATTENTION_GRU_ATTENTION_MODEL
- CNN_ATTENTION_BiLSTM_ATTENTION_MODEL
- CNN_ATTENTION_BiGRU_ATTENTION_MODEL
- CNN_ATTENTION_LSTM_MODEL
- CNN_ATTENTION_GRU_MODEL
- CNN_ATTENTION_BiLSTM_MODEL
- CNN_ATTENTION_BiGRU_MODEL

### Environment

- Python: 3.12.2
- PyTorch: 2.2.0
- CUDA: 12.1
- Conda: Latest version

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/mesabo/TimeSeriesForecasting.git
   ```

2. Download the UCI Household Power dataset text files into the `input/` directory.

3. Set up the environment with the required dependencies:

   ```bash
   conda create -n timeseriesforecasting python=3.12.2
   conda activate timeseriesforecasting
   pip install torch==2.2.0
   ```

4. If you have a GPU with Linux, you can use the provided `run.sh` script to execute the main program:

   ```bash
   cd root/src
   bash run.sh
   ```

   For other environments or running in an IDE, execute the `main.py` file located in the `src/` directory.

### Project Structure

The project has the following structure:

- `src/`: Contains the main codebase for model training, hyperparameter tuning, and evaluation.
    - `input_processing/`: Module for processing input data.
    - `hyperparameter_tuning/`: Module for hyperparameter tuning and study.
    - `output_processing/`: Module for handling model outputs, such as metrics and predictions.
- `input/`: Directory to store input datasets.
- `output/`: Directory to store model outputs, such as evaluation metrics and predictions.
- `utils/`: Contains utility functions and constants used across the project.

### Usage

To use this project for time series forecasting:

1. Prepare your dataset and place it in the `input/` directory.

2. Run the `main.py` file located in the `src/` directory. This script initiates the model training, hyperparameter
   tuning, and evaluation process.

3. Optionally, for GPU environments with Linux, you can use the provided `run.sh` script to execute the main program.

### Note

Ensure that you have the necessary dependencies installed and set up the environment correctly before running the code.
Additionally, customize the model selection and hyperparameters according to your specific forecasting requirements.

```plaintext
.
├── README.md
├── input
│   ├── data_features.json
│   ├── electricity
│   │   └── household_power_consumption.txt
│   └── household_power_consumption.txt
├── output
│   └── electricity
│       ├── hyperband
│       │   └── CNN-LSTM-Attention-based
│       │       └── 7_2_best_params.json
│       ├── losses
│       │   └── CNN-LSTM-Attention-based
│       │       ├── doc
│       │       │   └── 7_2_evaluation_losses.json
│       │       └── image
│       │           ├── 7_2_evaluation_metrics.png
│       │           └── 7_2_prediction.png
│       ├── metrics
│       │   └── CNN-LSTM-Attention-based
│       │       ├── doc
│       │       │   └── 7_2_evaluation_metrics.json
│       │       └── image
│       │           └── 7_2_evaluation_metrics.png
│       └── predictions
│           └── CNN-LSTM-Attention-based
│               └── image
│                   └── 7_2_prediction.png
├── requirements.txt
├── run.sh
├── sample.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── hyperparameter_tuning
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── build_best_model.py
│   │   ├── hyper_models.py
│   │   ├── model_tuner.py
│   │   └── model_tuner_study.py
│   ├── input_processing
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── data_processing.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── model_training.py
│   ├── output_processing
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── custom_functions.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       └── constants.py
└── time_serie_gpu.log
```

# Export install packages in the current environment

> pip list --format=freeze > requirements.txt

# Submit a code on gpu

> sbatch /home/user/path_to/TimeseriesForecasting/run-gpu1.sh

# Install tensorflow gpu

> pip install tensorflow[and-cuda]

# List running process

> watch -n 5 nvidia-smi

# Torch is availabnle here

> https://pytorch.org/get-started/locally/
