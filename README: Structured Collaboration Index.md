# README: Structured Collaboration Index

## Overview

The **calculate_SCI_scores** function computes the Structured Collaboration Index (SCI) scores for each team within a dataset. SCI scores are statistical measures that quantify the level of collaboration among team members based on their overlapping work hours.

This function analyzes the overlapping work schedules of team members to determine how closely they collaborate. It calculates an overall SCI score for each team and, if applicable, separate SCI scores for different collaboration modes within the team if the distribtion is bimodal.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Return Value](#return-value)
- [Detailed Explanation](#detailed-explanation)
- [Example](#example)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- **Overlap Calculation**: Computes the total overlapping hours between every pair of collaborators within a team.
- **Normalization**: Normalizes the overlap based on the minimum total hours worked by the collaborators.
- **SCI Score Computation**:
  - SCI_team: Calculates the overall SCI score for the team.
  - Bimodality Analysis: Checks for bimodality in the distribution of normalized overlaps using Gaussian Mixture Models.
  - SCI_ext and SCI_core: If the data is bimodal, computes separate SCI scores for the two modes.
- **Intersection Point Detection**: Finds the intersection point between two Gaussian components to separate the data into modes.

## Installation

Ensure you have the required Python packages installed:

```python
pip install numpy pandas scipy scikit-learn tqdm
```

## Usage

```python
results_df = calculate_SCI_scores(df)
```

**df**: A pandas DataFrame containing the necessary columns.

## Parameters

### df
- **Type**: pandas.DataFrame
- **Description**: The input DataFrame must contain at least the following columns:
  - `start`: Start datetime of each shift.
  - `end`: End datetime of each shift.
  - `Team`: Identifier for the team (integer type).
  - `collaborator_bk`: Unique identifier for each collaborator.

## Return Value

- **Type**: pandas.DataFrame
- **Description**: A DataFrame containing the SCI scores and additional information for each team.
- **Columns**:
  - `Team`: Team identifier.
  - `NumMembers`: Number of unique collaborators in the team.
  - `SCI_team`: The overall SCI score for the team.
  - `SCI_ext`: SCI score for the external mode (if applicable).
  - `SCI_core`: SCI score for the core mode (if applicable).
  - `ValleyPosition`: The position of the intersection point used to separate modes (if applicable).
  - `IsBimodal`: Boolean indicating whether the data was identified as bimodal.

## Detailed Explanation

The calculate_SCI_scores function operates through several key steps to compute the SCI scores for each team in the dataset:

1. **Data Preparation**

- **Datetime Conversion**: Ensures that the `start` and `end` columns are in datetime format using `pd.to_datetime`.
- **Team Identifier Conversion**: Converts the `Team` column to integer type for consistency.

```python
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])
df['Team'] = df['Team'].astype(int)
```

2. **Team Processing**

- **Team Iteration**: Processes each unique team in the DataFrame.
- **Collaborator Identification**: Collects all unique collaborators (`collaborator_bk`) within the team.
- **Total Hours Calculation**: Calculates the total hours worked by each collaborator.

```python
for team in tqdm(df['Team'].unique(), desc='Processing Teams', unit='team'):
    team_data = df[df['Team'] == team]
    collaborators = team_data['collaborator_bk'].unique()
    num_unique_members = len(collaborators)
    
    total_hours_per_collaborator = team_data.groupby('collaborator_bk').apply(
        lambda x: ((x['end'] - x['start']).sum().total_seconds() / 3600)
    )
```

3. **Overlap Calculation**

- **Pairwise Combinations**: Iterates over all possible pairs of collaborators within the team using `itertools.combinations`.
- **Overlap Computation**:
  - Uses the `calculate_overlap(df1, df2)` function to compute the total overlapping hours between two collaborators.
- **Algorithm**:
  - Converts the `start` and `end` times of both collaborators to NumPy arrays.
  - Computes the maximum of the start times and the minimum of the end times to find overlapping intervals.
  - Calculates the `overlap` duration in hours, setting negative overlaps to zero.
  - Sums all overlaps to get the total overlapping hours (`total_overlap`).

```python
def calculate_overlap(df1, df2):
    # Convert start and end times to NumPy arrays
    start1 = df1['start'].values.astype('datetime64[ns]')
    end1 = df1['end'].values.astype('datetime64[ns]')
    start2 = df2['start'].values.astype('datetime64[ns]')
    end2 = df2['end'].values.astype('datetime64[ns]')

    # Compute the maximum of the start times and the minimum of the end times
    latest_start = np.maximum(start1[:, None], start2)
    earliest_end = np.minimum(end1[:, None], end2)

    # Calculate the overlap in hours
    overlap = (earliest_end - latest_start) / np.timedelta64(1, 'h')
    overlap[overlap < 0] = 0  # Set negative overlaps to zero

    # Sum all overlaps to get the total overlap
    total_overlap = np.sum(overlap)

    return total_overlap
```

4. **Overlap Matrix Construction**

- **Matrix Initialization**: Creates a square matrix (DataFrame) where rows and columns represent collaborators.
- **Matrix Population**: Fills the `matrix` with normalized overlap values for each pair of collaborators.

```python
# Initialize the matrix for storing normalized overlap values
matrix = pd.DataFrame(0.0, index=collaborators, columns=collaborators)

# Calculate overlaps between all pairs of collaborators
for collaborator1, collaborator2 in combinations(collaborators, 2):
    df1 = team_data[team_data['collaborator_bk'] == collaborator1]
    df2 = team_data[team_data['collaborator_bk'] == collaborator2]

    # Calculate total overlap between two collaborators
    total_overlap = calculate_overlap(df1, df2)

    # Normalize the overlap
    max_collab_hours = min(
        total_hours_per_collaborator[collaborator1],
        total_hours_per_collaborator[collaborator2]
    )
    normalized_overlap = total_overlap / max_collab_hours if max_collab_hours > 0 else 0

    # Update the matrix
    matrix.loc[collaborator1, collaborator2] = normalized_overlap
    matrix.loc[collaborator2, collaborator1] = normalized_overlap
```

5. **Data Extraction for SCI Computation**

- **Upper Triangle Extraction**: Extracts the upper triangle of the overlap matrix (excluding the diagonal) to obtain all unique pairs without redundancy.
- **Data Filtering**: Filters out zero values to focus on actual overlaps, resulting in an array of non-zero normalized overlaps (`data_nonzero`).

```python
# Extract the upper triangle of the matrix, excluding the diagonal
upper_triangle_indices = np.triu_indices_from(matrix, k=1)
upper_triangle_values = matrix.values[upper_triangle_indices]

# Filter out zeros to focus on actual overlaps
data_nonzero = upper_triangle_values[upper_triangle_values > 0]
```

6. **SCI_team Computation**

- **SCI_team Calculation**:
  - Uses the `compute_SCI(mode_values)` function on data_nonzero to compute the overall SCI score for the team.
- **Algorithm**:
  - Calculates the mean and variance of the normalized overlap values.
  - Computes the `alpha` and `beta` parameters for a Beta distribution.
  - Calculates the `SCI` value as (alpha - beta) / (alpha + beta).

```python
def compute_SCI(mode_values):
    if len(mode_values) == 0:
        return np.nan

    mean = np.mean(mode_values)
    variance = np.var(mode_values, ddof=0)

    if variance == 0 or mean in [0, 1] or np.isnan(mean):
        return np.nan

    alpha = mean * ((mean * (1 - mean) / variance) - 1)
    beta = (1 - mean) * ((mean * (1 - mean) / variance) - 1)

    if np.isnan(alpha) or np.isnan(beta) or (alpha + beta) == 0:
        return np.nan

    SCI = (alpha - beta) / (alpha + beta)
    return SCI
```

```python
# Compute SCI_team
SCI_team = compute_SCI(data_nonzero)
```
7. **Bimodality Check**

- **Data Sufficiency**: Proceeds if there is more than one non-zero overlap value.
- **GMM Fitting**:
  - Fits *Gaussian Mixture Models (GMMs)* with one and two components to the data using [GaussianMixture](https://scikit-learn.org/stable/modules/mixture.html) from scikit-learn.
- **Model Selection**:
    - Computes the *Bayesian Information Criterion (BIC)* for both models.
    - If the BIC of the two-component model is lower than that of the one-component model, the data is considered bimodal.

```python
if len(data_nonzero) > 1:
    data_nonzero_reshaped = data_nonzero.reshape(-1, 1)

    # Fit GMMs with 1 and 2 components
    gmm1 = GaussianMixture(n_components=1)
    gmm2 = GaussianMixture(n_components=2)

    gmm1.fit(data_nonzero_reshaped)
    gmm2.fit(data_nonzero_reshaped)

    # Compute BIC scores
    bic1 = gmm1.bic(data_nonzero_reshaped)
    bic2 = gmm2.bic(data_nonzero_reshaped)

    # Select the model with the lower BIC
    if bic2 < bic1:
        is_bimodal = True
        # Proceed to find intersection and compute SCI_ext and SCI_core
```

8. **Mode Separation**

- **Intersection Point Detection**:
  - Uses the `find_gaussian_intersection(gmm, x_range)` function to find the `intersection` point (threshold) between the two Gaussian components.
- **Algorithm**:
  - Defines a `function gaussians_diff(x)` that computes the difference between the weighted log probabilities of the two Gaussian components at point x.
  - Uses *Brent’s method* ([brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)) to find the root of `gaussians_diff(x)`, which is the `intersection` point.

```python
def find_gaussian_intersection(gmm, x_range):
    def gaussians_diff(x):
        return (
            gmm.weights_[0] * gmm._estimate_weighted_log_prob(np.array([[x]]))[:, 0] -
            gmm.weights_[1] * gmm._estimate_weighted_log_prob(np.array([[x]]))[:, 1]
        )[0]

    try:
        intersection = brentq(gaussians_diff, x_range[0], x_range[1])
        return intersection
    except ValueError:
        return None

# Find intersection point
x_min, x_max = data_nonzero.min(), data_nonzero.max()
x_range = (x_min, x_max)
intersection = find_gaussian_intersection(gmm2, x_range)
```

- **Data Segmentation**:
  - Separates the normalized overlap values into two modes:
    - Mode 1 (`mode1_values`): Overlaps less than the threshold.
    - Mode 2 (`mode2_values`): Overlaps greater than or equal to the threshold.

```python
if intersection is not None and x_min < intersection < x_max:
    threshold = intersection
    valley_position = threshold

    # Separate data into mode 1 and mode 2 based on the threshold
    mode1_values = data_nonzero[data_nonzero < threshold]
    mode2_values = data_nonzero[data_nonzero >= threshold]
```

9. **SCI_ext and SCI_core Computation**

- **SCI_ext Calculation**:
  - Computes the SCI score for Mode 1 using `compute_SCI(mode1_values)`.

```python
# Calculate SCI_ext for values below the valley (mode 1)
SCI_ext = compute_SCI(mode1_values)
```

- **SCI_core Calculation**:
  - Computes the SCI score for Mode 2 using `compute_SCI(mode2_values)`.

```python
# Calculate SCI_core for values above the valley (mode 2)
SCI_core = compute_SCI(mode2_values)
```

10. **Result Compilation**

- **Data Aggregation**:
  - Collects the computed SCI scores and related information into a dictionary for each team.
```python
# Append the results for each team
results.append({
    'Team': team,
    'NumMembers': num_unique_members,
    'SCI_team': SCI_team,
    'SCI_ext': SCI_ext,
    'SCI_core': SCI_core,
    'ValleyPosition': valley_position,
    'IsBimodal': is_bimodal
})
```

- **DataFrame Conversion**:
  - Converts the list of dictionaries into a pandas DataFrame (`results_df`).

```python
# Convert the results into a DataFrame
results_df = pd.DataFrame(results)
```
- **Return**:
  - Returns the results_df containing the SCI scores and additional information for all teams.


## Example

```python
import pandas as pd
from datetime import datetime, timedelta

Sample data
data = {
    'start': [datetime(2023, 1, 1, 9, 0) + timedelta(hours=8*i) for i in range(10)],
    'end': [datetime(2023, 1, 1, 17, 0) + timedelta(hours=8*i) for i in range(10)],
    'Team': [1]*5 + [2]*5,
    'collaborator_bk': ['C1', 'C2', 'C3', 'C4', 'C5']*2,
}

df = pd.DataFrame(data)

# Calculate SCI scores
results_df = calculate_SCI_scores(df)

# Display the results
print(results_df)
```
**Sample Output**:
```
   Team  NumMembers  SCI_team  SCI_ext  SCI_core  ValleyPosition  IsBimodal
0     1           5       NaN      NaN       NaN             NaN      False
1     2           5       NaN      NaN       NaN             NaN      False
```
*Note: The sample data may not be sufficient to produce meaningful SCI scores. In practice, use a dataset with detailed and overlapping schedules to compute accurate SCI scores.*

## Dependencies

- **Python 3.x**
- **NumPy**
- **pandas**
- **SciPy**
- **scikit-learn**
- **tqdm**

## License

This project is licensed under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
