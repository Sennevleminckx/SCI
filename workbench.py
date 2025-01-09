import pandas as pd
from generate_schedule_lib import generate_schedule
from sci_lib import calculate_SCI_scores

# Define team details
team_details = {
    'start_date': datetime(2021, 1, 1),
    'end_date': datetime(2021, 3, 31),
    'teams': {
        1: {
            'MTS_size_fulltime': 15,
            'MTS_size_parttime': 5,
            'tasks': ["A", "B", "C"],
            'max_tasks_per_collaborator': 2,
            'freedom_level': .05  # Time is equally split among tasks
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
