import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_schedule(team_details):
    start_date, end_date = team_details['start_date'], team_details['end_date']
    all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    total_days = len(all_dates)

    def adjusted_max_days(proportion, total_days):
        return min(int(66 / proportion), total_days)

    # Reshuffle dates at the beginning
    np.random.shuffle(all_dates)

    schedule = []

    # Generate collaborator details
    for team_bk, team in team_details['teams'].items():
        team_size = team['team_size_fulltime'] + team['team_size_parttime']
        for i in range(1, team_size + 1):
            is_fulltime = i <= team['team_size_fulltime']
            proportion = 1 if is_fulltime else np.random.uniform(0.5, 0.9)
            max_days = adjusted_max_days(proportion, total_days)
            assigned_shifts = np.random.choice(team['shifts'], size=min(team['max_shifts_per_collaborator'], len(team['shifts'])), replace=False)

            collaborator_id = f"{team_bk}_{i}"
            work_dates = np.random.choice(all_dates, size=max_days, replace=False)

            shift_count = len(assigned_shifts)
            dates_per_shift = max_days // shift_count

            freedom_level = team.get('freedom_level', 1)  # Default to total freedom if not specified

            # Create a weighted preference for shifts
            for j, work_date in enumerate(work_dates):
                start_time = datetime.strptime(team['shifts'][j % len(team['shifts'])].split('-')[0], '%H%M')
                end_time = datetime.strptime(team['shifts'][j % len(team['shifts'])].split('-')[1], '%H%M')

                start_datetime = datetime.combine(work_date.date(), start_time.time())
                end_datetime = datetime.combine(work_date.date(), end_time.time())

                schedule.append({
                    'collaborator_bk': collaborator_id,
                    'Team': team_bk,
                    'start': start_datetime,
                    'end': end_datetime
                })

    schedule_df = pd.DataFrame(schedule)
    schedule_df = schedule_df.sort_values(by=['collaborator_bk', 'start']).reset_index(drop=True)

    return schedule_df


team_details = {
    'start_date': datetime(2021, 1, 1),
    'end_date': datetime(2021, 3, 31),
    'teams': {
        1: {'team_size_fulltime': 15, 'team_size_parttime': 0, 'shifts': ["0700-1400", "1401-2100", "2101-0659"], 'max_shifts_per_collaborator': 1, 'freedom_level': 0.5},
        2: {'team_size_fulltime': 20, 'team_size_parttime': 0, 'shifts': ["0700-1400", "1401-2100", "2101-0659"], 'max_shifts_per_collaborator': 1, 'freedom_level': 0.5},
...
}}

df = generate_schedule(team_details)
