import pandas as pd
import numpy as np
from python_speech_features import mfcc, delta, logfbank

# Load your CSV dataset into a Pandas DataFrame
df = pd.read_csv('emotions.csv')

# Assuming the last column is 'label' and other columns are features
features_columns = df.columns[:-1]

# Function to extract LPCC features from a row of features
def extract_lpcc(row):
    try:
        # Extract relevant features for the audio file
        audio_features = row[features_columns].values.astype(float)

        # Ensure the audio signal is 1D
        if audio_features.ndim > 1:
            audio_features = np.mean(audio_features, axis=1)

        # Extract MFCC features
        mfcc_features = mfcc(audio_features)

        # Compute LPCC features from MFCC features
        lpcc_features = delta(mfcc_features, 2)

        # Return the mean of LPCCs along each feature (column)
        return np.mean(lpcc_features, axis=0)

    except Exception as e:
        print(f"Error processing row: {e}")
        return None

# Apply the extract_lpcc function to each row in the dataset
df['lpcc'] = df.apply(extract_lpcc, axis=1)

# Drop rows where extraction failed
df = df.dropna()

# If you want to save the LPCC features to a new CSV file
df.to_csv('lpcc_dataset.csv', index=False)
