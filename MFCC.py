import pandas as pd
import librosa
import numpy as np

# Load  CSV dataset into a Pandas DataFrame
df = pd.read_csv('emotions.csv')

# Assuming the last column is 'label' and other columns are features
features_columns = df.columns[:-1]

# Function to extract MFCC features from a row of features
def extract_mfcc(row):
    try:
        # Extract relevant features for the audio file
        audio_features = row[features_columns].values.astype(float)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_features, sr=len(audio_features), n_mfcc=13)

        librosa.display.specshow(mfccs,x_axis='time')

        # Return the mean of MFCCs along each feature (column)
        return np.mean(mfccs, axis=1)

    except Exception as e:
        print(f"Error processing row: {e}")
        return None

# Apply the extract_mfcc function to each row in the dataset
df['mfcc'] = df.apply(extract_mfcc, axis=1)

# Drop rows where extraction failed
df = df.dropna()

# If you want to save the MFCC features to a new CSV file
df.to_csv('mfcc_dataset.csv', index=False)
