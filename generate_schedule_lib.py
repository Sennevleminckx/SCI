def generate_schedule(team_details, seed=None):
    """
    Generates a schedule for collaborators based on team details.

    Parameters:
    - team_details (dict): A dictionary containing team information, dates, tasks, etc.
    - seed (int, optional): An optional seed for random number generation to ensure reproducibility.

    Returns:
    - pandas.DataFrame: A DataFrame containing the generated schedule with start and end times.
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    # Set the random seed if provided for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Extract start and end dates from team_details
    start_date, end_date = team_details['start_date'], team_details['end_date']
    # Generate a list of all dates within the specified range
    all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    # Filter out weekends (assuming weekends are non-working days)
    all_dates = [date for date in all_dates if date.weekday() < 5]
    total_days = len(all_dates)

    def adjusted_max_days(proportion, total_days):
        """
        Adjusts the maximum number of working days based on the collaborator's work proportion.

        Parameters:
        - proportion (float): The proportion of full-time work (1 for full-time, <1 for part-time).
        - total_days (int): Total number of working days in the schedule.

        Returns:
        - int: The adjusted maximum number of working days.
        """
        return int(total_days * proportion)

    # Randomize the order of dates to distribute shifts more evenly
    np.random.shuffle(all_dates)
    schedule = []

    # Iterate over each team and their respective details
    for team_bk, team in team_details['teams'].items():
        # Calculate total team size
        team_size = team['MTS_size_fulltime'] + team['MTS_size_parttime']

        # Generate list of collaborator IDs
        collaborator_ids = [f"{team_bk}_{i}" for i in range(1, team_size + 1)]
        # Shuffle the collaborator IDs
        np.random.shuffle(collaborator_ids)

        tasks = team['tasks']
        num_tasks = len(tasks)
        collaborators_per_task = team_size // num_tasks
        extra_collaborators = team_size % num_tasks

        assigned_main_tasks = {}
        collaborator_index = 0

        # Assign main tasks to collaborators
        for task in tasks:
            num_collaborators_for_task = collaborators_per_task
            if extra_collaborators > 0:
                num_collaborators_for_task += 1
                extra_collaborators -= 1
            for _ in range(num_collaborators_for_task):
                if collaborator_index < len(collaborator_ids):
                    collaborator_id = collaborator_ids[collaborator_index]
                    assigned_main_tasks[collaborator_id] = task
                    collaborator_index += 1

        # Assign schedules to collaborators
        for collaborator_id in collaborator_ids:
            # Extract the index from collaborator_id
            i = int(collaborator_id.split('_')[1])
            # Determine if the collaborator is full-time or part-time
            is_fulltime = i <= team['MTS_size_fulltime']
            # Set work proportion based on employment status
            proportion = 1 if is_fulltime else np.random.uniform(0.5, 0.9)
            # Adjust maximum working days
            max_days = adjusted_max_days(proportion, total_days)
            # Max tasks per collaborator
            max_tasks = min(team['max_tasks_per_collaborator'], len(team['tasks']))

            # Main task is assigned from assigned_main_tasks
            main_task = assigned_main_tasks[collaborator_id]

            # Auxiliary tasks are the other tasks
            auxiliary_tasks = [t for t in tasks if t != main_task]
            if len(auxiliary_tasks) > 0 and max_tasks > 1:
                num_auxiliary_tasks = min(max_tasks - 1, len(auxiliary_tasks))
                assigned_auxiliary_tasks = np.random.choice(auxiliary_tasks, size=num_auxiliary_tasks, replace=False)
                assigned_tasks = [main_task] + list(assigned_auxiliary_tasks)
            else:
                assigned_tasks = [main_task]

            # Randomly select work dates for the collaborator
            work_dates = np.random.choice(all_dates, size=max_days, replace=False)
            # Get the freedom level for task preference
            freedom_level = team.get('freedom_level', 1)  # Default to 1 if not specified

            for work_date in work_dates:
                total_work_minutes = 8 * 60  # Total minutes in the workday (8 hours)
                N = len(assigned_tasks)

                if N == 1 or freedom_level == 0:
                    # If only one task or freedom_level=0, all time to main task
                    task_list = [{'Task': main_task, 'Duration': total_work_minutes}]
                else:
                    # Distribute time based on freedom_level
                    main_task_time_proportion = (1 - freedom_level) + (freedom_level / N)
                    auxiliary_task_time_proportion = freedom_level / N

                    # Calculate time allocations in minutes
                    main_task_time = int(total_work_minutes * main_task_time_proportion)
                    auxiliary_task_time_each = int(total_work_minutes * auxiliary_task_time_proportion)

                    # Adjust for rounding errors to ensure total time is 480 minutes
                    total_auxiliary_time = auxiliary_task_time_each * (N - 1)
                    main_task_time = total_work_minutes - total_auxiliary_time

                    # Create a list of tasks with their durations
                    task_list = [{'Task': main_task, 'Duration': main_task_time}]
                    for aux_task in assigned_tasks[1:]:
                        task_list.append({'Task': aux_task, 'Duration': auxiliary_task_time_each})

                # Remove any tasks with zero duration (if occurs due to rounding)
                task_list = [t for t in task_list if t['Duration'] > 0]

                # Randomize the order of tasks during the day
                np.random.shuffle(task_list)

                # Schedule the tasks sequentially from 09:00 onwards
                current_time = datetime.combine(work_date, datetime.strptime('09:00', '%H:%M').time())
                for t in task_list:
                    start_time = current_time
                    end_time = start_time + timedelta(minutes=t['Duration'])
                    # Ensure end time does not exceed 17:00
                    if end_time > datetime.combine(work_date, datetime.strptime('17:00', '%H:%M').time()):
                        end_time = datetime.combine(work_date, datetime.strptime('17:00', '%H:%M').time())
                    # Append to schedule
                    schedule.append({
                        'start': start_time,
                        'end': end_time,
                        'Team': team_bk,
                        'collaborator_bk': collaborator_id,
                        'Task': t['Task']
                    })
                    current_time = end_time  # Update current_time for next task

    # Create a DataFrame from the schedule
    schedule_df = pd.DataFrame(schedule)
    # Shuffle the schedule entries to randomize the order
    schedule_df = schedule_df.sample(frac=1).reset_index(drop=True)
    # Create a 'month' column representing the quarter
    schedule_df['month'] = schedule_df['start'].dt.to_period('Q').astype(str)

    return schedule_df
