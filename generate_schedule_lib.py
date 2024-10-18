import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def adjusted_max_days(proportion, total_days):
    """
    Adjusts the maximum number of working days based on the collaborator's work proportion.

    Parameters:
    - proportion (float): The proportion of full-time work (1 for full-time, <1 for part-time).
    - total_days (int): Total number of days in the schedule.

    Returns:
    - int: The adjusted maximum number of working days.
    """
    return min(int(66 * proportion), total_days)

def adjust_times(row):
    """
    Adjusts shift times, especially for shifts that go past midnight.

    Parameters:
    - row (Series): A pandas Series representing a row in a DataFrame.

    Returns:
    - Series: A Series containing adjusted 'start' and 'end' datetime objects.
    """
    # Parse start and end times
    start_time = datetime.strptime(row['timesheet_interval'] + row['start'], '%Y-%m-%d%H%M')
    end_time = datetime.strptime(row['timesheet_interval'] + row['end'], '%Y-%m-%d%H%M')

    # If the end time is earlier than the start time, the shift goes past midnight
    if end_time < start_time:
        end_time += timedelta(days=1)

    return pd.Series([start_time, end_time], index=['start', 'end'])

def generate_schedule(team_details, seed=None):
    """
    Generates a schedule for collaborators based on team details.

    Parameters:
    - team_details (dict): A dictionary containing team information, dates, shifts, etc.
    - seed (int, optional): An optional seed for random number generation to ensure reproducibility.

    Returns:
    - pandas.DataFrame: A DataFrame containing the generated schedule with start and end times.
    """
    # Set the random seed if provided for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Extract start and end dates from team_details
    start_date, end_date = team_details['start_date'], team_details['end_date']
    # Generate a list of all dates within the specified range
    all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    total_days = len(all_dates)

    # Randomize the order of dates to distribute shifts more evenly
    np.random.shuffle(all_dates)
    schedule = []

    # Iterate over each team and their respective details
    for team_bk, team in team_details['teams'].items():
        # Calculate total team size
        team_size = team['team_size_fulltime'] + team['team_size_parttime']
        # Iterate over each collaborator in the team
        for i in range(1, team_size + 1):
            # Determine if the collaborator is full-time or part-time
            is_fulltime = i <= team['team_size_fulltime']
            # Set work proportion based on employment status
            proportion = 1 if is_fulltime else np.random.uniform(0.5, 0.9)
            # Adjust maximum working days
            max_days = adjusted_max_days(proportion, total_days)
            # Assign shifts to the collaborator
            assigned_shifts = np.random.choice(
                team['shifts'],
                size=min(team['max_shifts_per_collaborator'], len(team['shifts'])),
                replace=False
            )
            # Generate a unique collaborator identifier
            collaborator_id = f"{team_bk}_{i}"
            # Randomly select work dates for the collaborator
            work_dates = np.random.choice(all_dates, size=max_days, replace=False)
            shift_count = len(assigned_shifts)
            # Get the freedom level for shift preference
            freedom_level = team.get('freedom_level', 1)  # Default to total freedom if not specified

            # Assign shifts to each work date
            for work_date in work_dates:
                if shift_count == 1:
                    # If only one shift, assign it directly
                    shift = assigned_shifts[0]
                else:
                    # Determine shift based on freedom level
                    if np.random.rand() < freedom_level:
                        shift = np.random.choice(assigned_shifts)
                    else:
                        shift = assigned_shifts[0]
                # Append the shift to the schedule
                schedule.append({
                    'activity_bk': shift,
                    'collaborator_bk': collaborator_id,
                    'timesheet_interval': work_date.strftime('%Y-%m-%d'),
                    'Team': team_bk
                })

    # Shuffle the schedule entries to randomize the order
    schedule_df = pd.DataFrame(schedule)
    schedule_df = schedule_df.sample(frac=1).reset_index(drop=True)

    # Keep only the necessary columns
    schedule_df = schedule_df[['timesheet_interval', 'activity_bk', 'collaborator_bk', 'Team']]

    # Filter out shifts where 'activity_bk' doesn't start with a digit
    schedule_df = schedule_df[schedule_df['activity_bk'].apply(lambda x: x[0].isdigit())]

    # Extract the first part of 'activity_bk' before any '/'
    schedule_df['activity_bk'] = schedule_df['activity_bk'].str.split('/').str[0]

    # Split 'activity_bk' into 'start' and 'end' times
    schedule_df[['start', 'end']] = schedule_df['activity_bk'].str.split('-', n=1, expand=True)

    # Apply time adjustments to handle overnight shifts
    schedule_df[['start', 'end']] = schedule_df.apply(adjust_times, axis=1)

    # Remove unnecessary columns after processing
    schedule_df.drop(['activity_bk', 'timesheet_interval'], axis=1, inplace=True)

    # Ensure 'start' and 'end' are datetime objects
    schedule_df['start'] = pd.to_datetime(schedule_df['start'])
    schedule_df['end'] = pd.to_datetime(schedule_df['end'])

    # Create a 'month' column representing the quarter
    schedule_df['month'] = schedule_df['start'].dt.to_period('Q').astype(str)

    return schedule_df
