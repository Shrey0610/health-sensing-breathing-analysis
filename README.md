# Scenario 2 - Health Sensing [25 Marks]
You have recently joined a healthcare startup called DeepMedico™ as a data scientist. Your team is working on detecting breathing irregularities that occur during sleep. As part of a pilot study, DeepMedico™ has collected overnight sleep data (8 hours per participant) from 5 subjects. The firm has provided you with the below dataset.

For each participant, the following physiological signals are available:
- Nasal Airflow (sampled at 32 Hz)
- Thoracic Movement (sampled at 32 Hz)
- SpO₂ (Oxygen Saturation) (sampled at 4 Hz)

In addition to these signals, the dataset also includes:
- An event file that contains annotations for breathing irregularities (e.g., apnea, hypopnea)
- A sleep profile file that records sleep stages over time

Your task is to analyze and model this data to detect abnormal breathing patterns during sleep.

## Understanding the Data and Visualization [3 Marks]
Your manager would like to explore how the recorded signals look across the 8-hour sleep sessions. Your task is to create clear visualizations of the data for each participant, making it easier to interpret patterns and irregularities.

Specifically, you are expected to:
- Plot the Nasal Airflow, Thoracic Movement, and SpO₂ signals over the entire 8-hour duration
- Overlay the annotated flow events (e.g., apnea, hypopnea) on top of the corresponding signal plots for visual reference
- Export the visualizations in PDF format, as strictly requested by your manager

**Deliverable:** Write a Python script named `vis.py` that accepts a folder path as input and generates a PDF visualization for that participant.

Example usage:
```bash
python vis.py -name "Data/AP20"
```

## Data Cleaning [4 Marks]
While reviewing the visualizations, one of your teammates notices that parts of the signal appear quite noisy. Your task is to clean the raw signals by filtering out high-frequency noise, making the data more suitable for further analysis and modeling.

**Hint:** Human breathing typically occurs at a rate of 10 to 24 breaths per minute (0.17 Hz to 0.4 Hz).

## Dataset Creation [8 Marks]
Create a dataset from the preprocessed signal data by:
- Splitting signals into 30-second windows with 50% overlap
- Labeling windows based on overlap with events (Hypopnea, Obstructive Apnea, or Normal)
- Saving the resulting dataset in an appropriate format

**Deliverable:** Write a Python script named `create_dataset.py`.

Example usage:
```bash
python create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

## Modeling [10 Marks]
Train models to classify breathing irregularities using:
- 1D Convolutional Neural Network (1D CNN)
- 1D Conv-LSTM

Use Leave-One-Participant-Out Cross-Validation and report comprehensive metrics including accuracy, precision, recall, sensitivity, specificity, and confusion matrices.

## Bonus: Sleep Stage Classification [5 Marks]
Extend your analysis to classify sleep stages using the sleep profile file, applying the same modeling approaches to predict Wake, REM, N1, N2, and N3 stages.