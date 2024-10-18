# README: Structured Collaboration Index
## Overview

This project provides tools for generating randomized work schedules for teams and calculating the **Structured Collaboration Index** (SCI) based on those schedules. The SCI is a statistical measure that quantifies the level of collaboration among team members by analyzing their overlapping work hours.

The project consists of two main components:

1. **`Schedule Generator`**: A function that creates randomized work schedules for collaborators across different teams, considering factors like full-time/part-time status, shift assignments, and shift preferences.
2. **`SCI Calculator`**: A function that computes SCI scores for each team by analyzing the overlapping work schedules, providing insights into collaboration patterns within teams.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Generating Schedules](#1-generating-schedules)
  - [2. Calculating SCI Scores](#2-calculating-sci-scores)
- [Example](#examples)
- [Dependencies](#dependencies)
- [License](#license)
- [Detailed Explanation of Schedule Generation](#detailed-explanation-of-schedule-generation)
- [Detailed Explanation of SCI Calculation](#detailed-explanation-of-sci-calculation)
- [Additional Notes](#additional-notes)

## Features
- **Randomized Scheduling**: Generates work schedules with customizable parameters for teams and collaborators.
- **Full-Time and Part-Time Support**: Adjusts working days and hours based on employment status.
- **Shift Preferences**: Allows for varying degrees of randomness in shift assignments.
- **Overlap Calculation**: Computes total overlapping hours between collaborators.
- **SCI Computation**: Calculates overall SCI scores and separate scores for different collaboration modes within teams.
- **Gaussian Mixture Modeling**: Uses Gaussian Mixture Models to identify collaboration modes.


## Installation
Ensure you have **Python 3.x** installed along with the required packages. You can install the dependencies using `pip`:

```python
pip install numpy pandas scipy scikit-learn tqdm
```

## Project Structure
- **generate_schedule_lib.py**: Contains the `generate_schedule` function and helper functions for schedule generation.
- **sci_lib.py**: Contains the `calculate_SCI_scores` function and helper functions for SCI computation.
- **workbench.py**: Script demonstrating how to use the libraries.
- **README.md**: Project documentation (this file).

## Usage

### 1. Generating Schedules
#### Importing the function
```python
from generate_schedule_lib import generate_schedule
```
#### Defining Team Details
Create a dictionary containing team configurations and scheduling parameters.
```python
import pandas as pd
from datetime import datetime

team_details = {
    'start_date': datetime(2023, 1, 1),
    'end_date': datetime(2023, 3, 31),
    'teams': {
        'TeamAlpha': {
            'team_size_fulltime': 3,
            'team_size_parttime': 2,
            'shifts': ['0800-1600', '1600-0000'],
            'max_shifts_per_collaborator': 2,
            'freedom_level': 0.7  # Optional
        },
        'TeamBeta': {
            'team_size_fulltime': 4,
            'team_size_parttime': 1,
            'shifts': ['0900-1700', '1700-0100'],
            'max_shifts_per_collaborator': 1
            # freedom_level defaults to 1 if not specified
        }
    }
}
```
#### Generating the Schedule
```python
schedule_df = generate_schedule(team_details, seed=42)
```
*note: Due to the random elements in schedule generation, results may vary between runs unless a `seed` is set (seed is optional, Default = None).*

#### Understanding the output
- schedule_df is a pandas DataFrame containing:
  - `collaborator_bk`: Unique identifier for each collaborator.
  - `Team`: Team identifier.
  - `start`: Start datetime of the shift.
  - `end`: End datetime of the shift.
    
### 2. Calculating SCI Scores
#### Importing the function
```python
from sci_lib import calculate_SCI_scores
```
#### Calculating the SCI scores

```python
results_df = calculate_SCI_scores(df)
```

**df**: A pandas DataFrame containing the necessary column:
- `start`: Start datetime of each shift.
- `end`: End datetime of each shift.
- `Team`: Identifier for the team (integer type).
- `collaborator_bk`: Unique identifier for each collaborator.

#### Understanding the Output:
- **Type**: pandas.DataFrame
- **Description**: A DataFrame containing the SCI scores and additional information for each team.
- **Columns**:
  - `Team`: Team identifier.
  - `NumMembers`: Number of unique collaborators in the team.
  - `SCI_team`: The overall SCI score for the team.
  - `SCI_ext`: SCI score for the extended mode.
  - `SCI_core`: SCI score for the core mode.
  - `ValleyPosition`: The position of the intersection point used to separate modes.

## Examples
### Complete Workflow Example:
```python
import pandas as pd
from datetime import datetime
from generate_schedule_lib import generate_schedule
from sci_lib import calculate_SCI_scores

# Define team details
team_details = {
    'start_date': datetime(2023, 1, 1),
    'end_date': datetime(2023, 3, 31),
    'teams': {
        'TeamAlpha': {
            'team_size_fulltime': 3,
            'team_size_parttime': 2,
            'shifts': ['0800-1600', '1600-0000'],
            'max_shifts_per_collaborator': 2,
            'freedom_level': 0.7
        },
        'TeamBeta': {
            'team_size_fulltime': 4,
            'team_size_parttime': 1,
            'shifts': ['0900-1700', '1700-0100'],
            'max_shifts_per_collaborator': 1
        }
    }
}

# Generate the schedule
schedule_df = generate_schedule(team_details, seed=42)

# Calculate SCI scores
sci_scores_df = calculate_SCI_scores(schedule_df)

# Display the results
print("Schedule DataFrame:")
print(schedule_df.head())
print("\nSCI Scores DataFrame:")
print(sci_scores_df)
```
### Sample Output:
```python
Schedule DataFrame:
  collaborator_bk       Team               start                 end   month
0   TeamAlpha_2  TeamAlpha 2023-01-02 08:00:00 2023-01-02 16:00:00  2023Q1
1     TeamBeta_1   TeamBeta 2023-01-02 09:00:00 2023-01-02 17:00:00  2023Q1
2   TeamAlpha_3  TeamAlpha 2023-01-03 16:00:00 2023-01-04 00:00:00  2023Q1
3     TeamBeta_2   TeamBeta 2023-01-03 09:00:00 2023-01-03 17:00:00  2023Q1
4   TeamAlpha_1  TeamAlpha 2023-01-04 08:00:00 2023-01-04 16:00:00  2023Q1

SCI Scores DataFrame:
        Team  NumMembers  SCI_team  SCI_ext  SCI_core  ValleyPosition  IsBimodal
0  TeamAlpha           5  0.126633      NaN       NaN             NaN      False
1   TeamBeta           5  0.204586      NaN       NaN             NaN      False
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

## Detailed Explanation of Schedule Generation

1. **Random Seed Initialization**: If a `seed` is provided, the random number generator is initialized with it for reproducibility.

2. **Date Generation**: Generates a list of all dates within the specified `start_date` and `end_date`.

3. **Date Shuffling**: Randomizes the order of dates to evenly distribute shifts across the schedule.

4. **Schedule Generation**:
   - **Team Iteration**: Iterates over each team in `team_details`.
   - **Collaborator Determination**: Calculates the total number of collaborators and determines their employment status (full-time or part-time).
   - **Working Days Adjustment**: Adjusts the maximum number of working days based on whether the collaborator is full-time or part-time.
   - **Shift Assignment**:
     - **Shift Selection**: Assigns shifts to collaborators based on `max_shifts_per_collaborator`.
     - **Freedom Level Application**: Determines shift assignment randomness using `freedom_level`.
   - **Work Date Selection**: Randomly selects dates for each collaborator's shifts.

5. **Data Processing**:
   - **Filtering**: Removes any shifts that do not start with a digit (ensuring valid shift formats).
   - **Time Adjustment**: Splits shift strings into start and end times and adjusts for overnight shifts.
   - **Datetime Conversion**: Converts start and end times into datetime objects.
   - **Quarter Calculation**: Adds a column to represent the quarter of the year.


## Detailed Explanation of SCI calculation
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
    """
    Calculate the total overlap in hours between two collaborators.

    Parameters:
    - df1 (DataFrame): DataFrame for collaborator 1 with 'start' and 'end' columns.
    - df2 (DataFrame): DataFrame for collaborator 2 with 'start' and 'end' columns.

    Returns:
    - total_overlap (float): The total overlap in hours.
    """
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
- **Normalization**:
  - The overlap is normalized from each collaborator’s perspective separately, resulting in an asymmetric matrix.  
- **Matrix Population**: Fills the `matrix` with normalized overlap values for each pair of collaborators.

```python
# Initialize the matrix for storing normalized overlap values
matrix = pd.DataFrame(0.0, index=collaborators, columns=collaborators)

# Calculate overlaps between all pairs of collaborators
for collaborator1, collaborator2 in combinations(collaborators, 2):
    # [Overlap calculation]
    # Normalize the overlap from each collaborator's perspective
    normalized_overlap_1 = total_overlap / total_hours_collaborator1 if total_hours_collaborator1 > 0 else 0
    normalized_overlap_2 = total_overlap / total_hours_collaborator2 if total_hours_collaborator2 > 0 else 0

    # Update the matrix with asymmetric values
    matrix.loc[collaborator1, collaborator2] = normalized_overlap_1
    matrix.loc[collaborator2, collaborator1] = normalized_overlap_2
```

5. **Data Extraction for SCI Computation**

- **Matrix Stacking**: Stacks the asymmetric matrix to get all pairwise overlaps.
- **Removing Self-Overlaps**: Removes diagonal elements (self-overlaps).
- **Data Filtering**: Filters out zero values to focus on actual overlaps, resulting in an array of non-zero normalized overlaps (`data_nonzero`).

```python
# Stack the matrix to get all pairwise overlaps
stacked_matrix = matrix.stack()

# Remove self-overlaps (diagonal elements)
data_off_diagonal = stacked_matrix[stacked_matrix.index.get_level_values(0) != stacked_matrix.index.get_level_values(1)]

# Get the overlap values
data_values = data_off_diagonal.values

# Filter out zeros to focus on actual overlaps
data_nonzero = data_values[data_values > 0]
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
7. **Gaussian Mixture Modeling**
- **GMM Fitting**:
  - Fits *Gaussian Mixture Models (GMMs)* with two components to the data using [GaussianMixture](https://scikit-learn.org/stable/modules/mixture.html) from scikit-learn.
- **Intersection Point Detection**:
  - Uses the `find_gaussian_intersection(gmm, x_range)` function to find the `intersection` point (threshold) between the two Gaussian components.
  - If unsuccessful, uses the mean of the data as the threshold.
- **Algorithm**:
  - Uses the `function gaussians_diff(x)` that computes the difference between the weighted log probabilities of the two Gaussian components at point x.
  - Uses *Brent’s method* ([brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)) to find the root of `gaussians_diff(x)`, which is the `intersection` point.

```python
def find_gaussian_intersection(gmm, x_range):
    """
    Find the intersection point between two Gaussian components in a GMM.

    Parameters:
    - gmm (GaussianMixture): The fitted Gaussian Mixture Model with two components.
    - x_range (tuple): The range of x values to search for the intersection.

    Returns:
    - intersection (float): The x-value where the two Gaussians intersect.
    """
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
```

```python
# Proceed to fit GMM with 2 components
data_nonzero_reshaped = data_nonzero.reshape(-1, 1)

# Fit GMM with 2 components
gmm2 = GaussianMixture(n_components=2)
gmm2.fit(data_nonzero_reshaped)

# Find intersection point between the two Gaussians
x_min, x_max = data_nonzero.min(), data_nonzero.max()
x_range = (x_min, x_max)
intersection = find_gaussian_intersection(gmm2, x_range)

if intersection is not None and x_min < intersection < x_max:
    threshold = intersection
else:
    # Unable to find a valid intersection, use mean as threshold
    threshold = data_nonzero.mean()

valley_position = threshold
```

8. **Mode Separation**

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


## Additional Notes

- **Normalization of Overlaps**: Overlaps are normalized from each collaborator’s perspective separately, resulting in an asymmetric overlap matrix.
- **Gaussian Mixture Modeling**: The function fits a GMM with two components regardless of the data distribution, aiming to identify potential collaboration modes.
- **Threshold Determination**: If the intersection point between the two Gaussians cannot be found, the mean of the data is used as the threshold for mode separation.
- **Data Requirements**: Ensure that your DataFrame contains sufficient overlapping schedules to compute meaningful SCI scores.
- **Performance Considerations**: Processing large datasets may take time; progress is displayed using `tqdm`.
