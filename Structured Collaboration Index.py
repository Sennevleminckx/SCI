import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm

def calculate_overlap(df1, df2):
    """
    This method calculates the overlap between two datasets.

    Args:
        * param df1: The first dataset (pandas DataFrame) containing 'start' and 'end' columns.
        * param df2: The second dataset (pandas DataFrame) containing 'start' and 'end' columns.
    Returns:
        * The overlap (numpy array) between the two datasets in hours.
    """
    latest_start = np.maximum(df1['start'].values[:, None], df2['start'].values)
    earliest_end = np.minimum(df1['end'].values[:, None], df2['end'].values)
    overlap = (earliest_end - latest_start) / np.timedelta64(1, 'h')
    overlap[overlap < 0] = 0
    return overlap


df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])

# Initialize a list to store the results
results = []

# Process each team
for team in tqdm(df['Team'].unique(), desc='Processing Teams', unit='team'):
    team_data = df[df['Team'] == team]
    collaborators = team_data['collaborator_bk'].unique()
    num_unique_members = len(collaborators)

    # Calculate total hours per collaborator
    total_hours_per_collaborator = team_data.groupby('collaborator_bk').apply(
        lambda x: ((x['end'] - x['start']).sum().total_seconds() / 3600))

    # Initialize the matrix for storing overlap values
    matrix = pd.DataFrame(0.0, index=collaborators, columns=collaborators)

    for collaborator1, collaborator2 in combinations(collaborators, 2):
        df1 = team_data[team_data['collaborator_bk'] == collaborator1]
        df2 = team_data[team_data['collaborator_bk'] == collaborator2]

        overlap_matrix = calculate_overlap(df1, df2)
        total_overlap = overlap_matrix.sum()

        max_collab_hours = min(total_hours_per_collaborator[collaborator1], total_hours_per_collaborator[collaborator2])
        normalized_overlap = total_overlap / max_collab_hours if max_collab_hours > 0 else 0

        matrix.loc[collaborator1, collaborator2] = normalized_overlap
        matrix.loc[collaborator2, collaborator1] = normalized_overlap

    # Flatten the matrix to a 1D array and filter out zeros
    data = matrix.values.flatten()
    data_nonzero = data[data > 0]

    # Calculate the sample mean and variance
    mean = np.mean(data_nonzero)
    variance = np.var(data_nonzero, ddof=0)  # ddof=0 for population variance want we hebben het hele team. ddof = 1 voor sample

    # Estimate α and β using the method of moments, avoiding division by zero
    if variance == 0 or mean == 1 or mean == 0:
        alpha = np.nan
        beta = np.nan
    else:
        alpha = mean * ((mean * (1 - mean) / variance) - 1)
        beta = (1 - mean) * ((mean * (1 - mean) / variance) - 1)

    # Append the results
    results.append({'Team': team,'Alpha': alpha, 'Beta': beta})

# Convert the results into a DataFrame
results_df = pd.DataFrame(results)
results_df['SCI'] = (results_df['Alpha'] - results_df['Beta']) / (results_df['Alpha'] + results_df['Beta'])
results_df
