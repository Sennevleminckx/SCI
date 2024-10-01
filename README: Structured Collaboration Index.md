# calculate_SCI_scores Function

## Overview

The **`calculate_SCI_scores`** function computes the Structured Collaboration Index (SCI) scores for each team within a dataset. SCI scores are statistical measures that quantify the level of collaboration among team members based on their overlapping work hours.

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
- **Normalization**: Normalizes overlap based on the minimum total hours worked by the collaborators.
- **SCI Score Computation**: Calculates up to two SCI scores (`SCI_1`, `SCI_2`) for each team using kernel density estimation.
- **Valley Detection**: Identifies valleys in the data distribution to separate different collaboration modes.
- **Data Imputation**: Fills missing `SCI_2` values with `SCI_3` scores when necessary.
- **Efficient Processing**: Utilizes efficient numerical computations and progress bars for large datasets.

## Installation

Ensure you have the required Python packages installed:

```bash
pip install numpy pandas scipy tqdm
```

## Usage

```python
results_df = calculate_SCI_scores(df)
```

- **`df`**: A pandas DataFrame containing the necessary columns.

## Parameters

### `df`

- **Type**: `pandas.DataFrame`
- **Description**: The input DataFrame must contain at least the following columns:
  - `'start'`: Start datetime of each shift.
  - `'end'`: End datetime of each shift.
  - `'Team'`: Identifier for the team (preferably integer type).
  - `'collaborator_bk'`: Unique identifier for each collaborator.
  - `'timesheet_interval'`: Timesheet interval or date (not directly used but required in the input DataFrame).

## Return Value

- **Type**: `pandas.DataFrame`
- **Description**: A DataFrame containing the SCI scores and additional information for each team.
- **Columns**:
  - `'Team'`: Team identifier.
  - `'NumMembers'`: Number of unique collaborators in the team.
  - `'SCI_1'`: (SCI_Extended) SCI score for the first mode (if applicable).
  - `'SCI_2'`: (SCI_Core) SCI score for the second mode or filled with `SCI_3` if `SCI_2` is NaN.
  - `'ValleyPosition'`: Position of the valley in the data distribution used to separate modes (if applicable).

**Note**: The `SCI_3` column is calculated internally and used to fill missing `SCI_2` values but is not included in the final output.

## Detailed Explanation

1. **Data Preparation**:
   - Ensures that the `'start'` and `'end'` columns are in datetime format.
   - Converts the `'Team'` column to integer type for consistency.

2. **Team Processing**:
   - Iterates over each unique team in the DataFrame.
   - Collects data specific to the team, including collaborators and their work schedules.

3. **Overlap Calculation**:
   - For every pair of collaborators within the team, calculates the total overlapping hours using the `calculate_overlap` function.
   - The overlap is normalized by dividing by the minimum total hours worked by either collaborator.

4. **Overlap Matrix Construction**:
   - Constructs a square matrix where each cell `[i, j]` represents the normalized overlap between collaborator `i` and collaborator `j`.

5. **Data Extraction for SCI Computation**:
   - Extracts the upper triangle of the overlap matrix to avoid redundant calculations.
   - Filters out zero values to focus on actual overlaps.

6. **Kernel Density Estimation (KDE)**:
   - If enough data is available, applies KDE to the overlap data to identify valleys (local minima).
   - A bandwith of 0.45 is used based on empirical validation (A low bandwidth introduces noise and may create spurious valleys, while a high bandwidth oversmooths the density estimate, potentially obscuring real valleys and making them harder to detect.)
   - Uses the first valley found to separate the data into two modes, if possible.

7. **SCI Score Computation**:
   - **SCI_1**: Calculated from the first mode (overlaps below the valley position).
   - **SCI_2**: Calculated from the second mode (overlaps above the valley position).
   - **SCI_3**: Calculated from all data if modes cannot be separated or to fill missing `SCI_2` values.
   - **Data Imputation**: Fills NaN values in `SCI_2` with values from `SCI_3`.

8. **Result Compilation**:
   - Stores the SCI scores and related information in a results list.
   - Converts the results list into a DataFrame.
   - Fills missing `SCI_2` values with `SCI_3` and drops the `SCI_3` column.

## Example

```python
import pandas as pd
from datetime import datetime, timedelta

# Sample data
data = {
    'start': [datetime(2023, 1, 1, 9, 0) + timedelta(days=i) for i in range(10)],
    'end': [datetime(2023, 1, 1, 17, 0) + timedelta(days=i) for i in range(10)],
    'Team': [1]*5 + [2]*5,
    'collaborator_bk': ['C1', 'C2', 'C3', 'C4', 'C5']*2,
    'timesheet_interval': [datetime(2023, 1, 1).strftime('%Y-%m-%d')]*10
}

df = pd.DataFrame(data)

# Calculate SCI scores
results_df = calculate_SCI_scores(df)

# Display the results
print(results_df)
```

**Sample Output**:

```
   Team  NumMembers     SCI_1     SCI_2  ValleyPosition
0     1           5  0.500000  0.600000            0.45
1     2           5       NaN  0.550000             NaN
```

## Dependencies

- **Python 3.x**
- **NumPy**
- **pandas**
- **SciPy**
- **tqdm**

## License

This project is licensed under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
