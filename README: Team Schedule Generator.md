# Schedule Generator Function

## Overview

The **Schedule Generator** is a Python function designed to create randomized work schedules for collaborators across different teams. It accounts for various parameters such as full-time and part-time staff, shift assignments, maximum shifts per collaborator, and the level of freedom in shift selection.

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

- **Randomized Scheduling**: Generates a random schedule while adhering to specified constraints.
- **Full-Time and Part-Time Support**: Adjusts working days based on employment status.
- **Shift Preferences**: Assigns shifts based on the collaborator's assigned shifts and freedom level.
- **Date Range Flexibility**: Allows scheduling over any given date range.
- **Data Processing**: Cleans and processes data to provide start and end times for each shift.

## Installation

No special installation is required beyond having the necessary Python packages installed.

## Usage

```python
schedule_df = generate_schedule(team_details, seed=None)
```

- **`team_details`**: A dictionary containing team configurations and scheduling parameters.
- **`seed`** *(optional)*: An integer to set the random seed for reproducibility.

## Parameters

### `team_details`

A dictionary with the following structure:

```python
team_details = {
    'start_date': datetime object,
    'end_date': datetime object,
    'teams': {
        'team_identifier': {
            'team_size_fulltime': int,
            'team_size_parttime': int,
            'shifts': list of shift strings (e.g., ['0800-1600', '1600-0000']),
            'max_shifts_per_collaborator': int,
            'freedom_level': float between 0 and 1 (optional)
        },
        # Add more teams as needed
    }
}
```

- **`start_date`**: The start date for scheduling (datetime object).
- **`end_date`**: The end date for scheduling (datetime object).
- **`teams`**: A dictionary of team configurations.

#### Team Configuration Parameters

- **`team_size_fulltime`**: Number of full-time collaborators.
- **`team_size_parttime`**: Number of part-time collaborators.
- **`shifts`**: A list of shift strings in 'HHMM-HHMM' format.
- **`max_shifts_per_collaborator`**: Maximum number of different shifts a collaborator can have.
- **`freedom_level`** *(optional)*: A float between 0 and 1 indicating the randomness in shift assignment.
  - **Default**: `1` (maximum freedom).

### `seed`

- **Type**: `int`
- **Description**: An optional seed for random number generation to ensure the same schedule is produced every time.
- **Default**: `None`

## Return Value

The function returns a `pandas.DataFrame` with the following columns:

- **`collaborator_bk`**: Unique identifier for the collaborator.
- **`Team`**: Team identifier.
- **`start`**: Start datetime of the shift.
- **`end`**: End datetime of the shift.
- **`month`**: The quarter (e.g., '2023Q1') in which the shift occurs.

## Detailed Explanation

1. **Random Seed Initialization**: If a seed is provided, the random number generator is initialized with it for reproducibility.

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

## Example

```python
from datetime import datetime
import pandas as pd

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
            # freedom_level defaults to 1
        }
    }
}

# Generate the schedule
schedule_df = generate_schedule(team_details, seed=123)

# Display the first few rows of the schedule
print(schedule_df.head())
```

**Sample Output**:

```
  collaborator_bk       Team               start                 end   month
0     TeamBeta_1   TeamBeta 2023-01-15 09:00:00 2023-01-15 17:00:00  2023Q1
1   TeamAlpha_2  TeamAlpha 2023-02-10 16:00:00 2023-02-11 00:00:00  2023Q1
2   TeamAlpha_4  TeamAlpha 2023-03-05 08:00:00 2023-03-05 16:00:00  2023Q1
3     TeamBeta_3   TeamBeta 2023-01-20 09:00:00 2023-01-20 17:00:00  2023Q1
4   TeamAlpha_5  TeamAlpha 2023-02-25 08:00:00 2023-02-25 16:00:00  2023Q1
```

## Dependencies

- **Python 3.x**
- **NumPy**
- **pandas**

## License
This project is licensed under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
