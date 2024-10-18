import pandas as pd
from generate_schedule_lib import generate_schedule
from structured_collaboration_index_lib import calculate_SCI_scores

# Define team details
team_details = {
    'start_date': pd.to_datetime('2023-01-01'),
    'end_date': pd.to_datetime('2023-03-31'),
    'teams': {
        1: {
            'team_size_fulltime': 5,
            'team_size_parttime': 3,
            'shifts': ['0800-1600', '1600-0000', '0000-0800'],
            'max_shifts_per_collaborator': 2,
            'freedom_level': 0.8
        },
        # Add more teams as needed
    }
}

# Generate the schedule
schedule_df = generate_schedule(team_details, seed=42)

# Calculate SCI scores
sci_scores_df = calculate_SCI_scores(schedule_df)

# Display the results
print(sci_scores_df)
