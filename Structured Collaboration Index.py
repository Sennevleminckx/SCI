def calculate_SCI_scores(df):
    """
    Calculate Structured Collaboration Index (SCI) scores for each team in the dataframe.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing at least the following columns:
        'start', 'end', 'Team', 'collaborator_bk', 'timesheet_interval'

    Returns:
    - pandas.DataFrame: A dataframe containing SCI scores and additional information for each team.
    """
    import numpy as np
    import pandas as pd
    from itertools import combinations
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    from tqdm import tqdm
    import warnings

    # Suppress specific FutureWarning
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="use_inf_as_na option is deprecated and will be removed in a future version"
    )

    # Define the function to calculate total overlap between two collaborators
    def calculate_overlap(df1, df2):
        """
        Calculate the total overlap in hours between two collaborators.

        Parameters:
        - df1 (pandas.DataFrame): DataFrame for collaborator 1 with 'start' and 'end' columns.
        - df2 (pandas.DataFrame): DataFrame for collaborator 2 with 'start' and 'end' columns.

        Returns:
        - float: The total overlap in hours.
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

    # Ensure time columns are in datetime format
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['Team'] = df['Team'].astype(int)

    # Initialize list to collect results
    results = []

    # Process each team individually
    for team in tqdm(df['Team'].unique(), desc='Processing Teams', unit='team'):
        team_data = df[df['Team'] == team]
        collaborators = team_data['collaborator_bk'].unique()
        num_unique_members = len(collaborators)

        # Calculate total hours per collaborator
        total_hours_per_collaborator = team_data.groupby('collaborator_bk').apply(
            lambda x: ((x['end'] - x['start']).sum().total_seconds() / 3600)
        )

        # Initialize a matrix for storing normalized overlap values
        matrix = pd.DataFrame(0.0, index=collaborators, columns=collaborators)

        # Calculate overlaps between all pairs of collaborators
        for collaborator1, collaborator2 in combinations(collaborators, 2):
            df1 = team_data[team_data['collaborator_bk'] == collaborator1]
            df2 = team_data[team_data['collaborator_bk'] == collaborator2]

            # Calculate total overlap between two collaborators
            total_overlap = calculate_overlap(df1, df2)

            # Normalize the overlap based on the minimum total hours of the two collaborators
            max_collab_hours = min(
                total_hours_per_collaborator[collaborator1],
                total_hours_per_collaborator[collaborator2]
            )
            normalized_overlap = total_overlap / max_collab_hours if max_collab_hours > 0 else 0

            # Update the matrix with the normalized overlap
            matrix.loc[collaborator1, collaborator2] = normalized_overlap
            matrix.loc[collaborator2, collaborator1] = normalized_overlap

        # Extract the upper triangle of the matrix, excluding the diagonal
        upper_triangle_indices = np.triu_indices_from(matrix, k=1)
        upper_triangle_values = matrix.values[upper_triangle_indices]

        # Filter out zeros to focus on actual overlaps
        data_nonzero = upper_triangle_values[upper_triangle_values > 0]

        # Initialize variables for SCI values
        SCI_1 = SCI_2 = SCI_3 = np.nan
        valley_position = None

        # Compute KDE to identify valleys if enough data is available
        if len(data_nonzero) > 1:
            bandwidth = 0.45  # Adjust bandwidth if needed
            kde = gaussian_kde(data_nonzero, bw_method=bandwidth)
            x_vals = np.linspace(0, 1, 1000)
            kde_vals = kde(x_vals)

            # Identifying valleys (local minima) in the KDE
            valleys, _ = find_peaks(-kde_vals)

            # Determine threshold using the first valley found, if any
            if len(valleys) > 0:
                threshold = x_vals[valleys[0]]  # Use the first valley as the threshold
                valley_position = threshold

                # Separate data into mode 1 and mode 2 based on the threshold
                mode1_values = data_nonzero[data_nonzero < threshold]
                mode2_values = data_nonzero[data_nonzero >= threshold]

                # Calculate SCI_1 for values below the valley (mode 1)
                mean1 = np.mean(mode1_values) if len(mode1_values) > 0 else np.nan
                variance1 = np.var(mode1_values, ddof=0) if len(mode1_values) > 0 else np.nan
                if variance1 == 0 or mean1 in [0, 1] or np.isnan(mean1):
                    alpha1 = beta1 = np.nan
                else:
                    alpha1 = mean1 * ((mean1 * (1 - mean1) / variance1) - 1)
                    beta1 = (1 - mean1) * ((mean1 * (1 - mean1) / variance1) - 1)
                SCI_1 = (alpha1 - beta1) / (alpha1 + beta1) if not np.isnan([alpha1, beta1]).any() and (alpha1 + beta1) != 0 else np.nan

                # Calculate SCI_2 for values above the valley (mode 2)
                mean2 = np.mean(mode2_values) if len(mode2_values) > 0 else np.nan
                variance2 = np.var(mode2_values, ddof=0) if len(mode2_values) > 0 else np.nan
                if variance2 == 0 or mean2 in [0, 1] or np.isnan(mean2):
                    alpha2 = beta2 = np.nan
                else:
                    alpha2 = mean2 * ((mean2 * (1 - mean2) / variance2) - 1)
                    beta2 = (1 - mean2) * ((mean2 * (1 - mean2) / variance2) - 1)
                SCI_2 = (alpha2 - beta2) / (alpha2 + beta2) if not np.isnan([alpha2, beta2]).any() and (alpha2 + beta2) != 0 else np.nan

                # Transfer to SCI_3 if only one SCI is valid
                if np.isnan(SCI_1) and not np.isnan(SCI_2):
                    SCI_3 = SCI_2
                    SCI_2 = np.nan
                elif not np.isnan(SCI_1) and np.isnan(SCI_2):
                    SCI_3 = SCI_1
                    SCI_1 = np.nan

            else:
                # If no valley is found, include all values and calculate SCI_3
                mode3_values = data_nonzero
                mean3 = np.mean(mode3_values) if len(mode3_values) > 0 else np.nan
                variance3 = np.var(mode3_values, ddof=0) if len(mode3_values) > 0 else np.nan
                if variance3 == 0 or mean3 in [0, 1] or np.isnan(mean3):
                    alpha3 = beta3 = np.nan
                else:
                    alpha3 = mean3 * ((mean3 * (1 - mean3) / variance3) - 1)
                    beta3 = (1 - mean3) * ((mean3 * (1 - mean3) / variance3) - 1)
                SCI_3 = (alpha3 - beta3) / (alpha3 + beta3) if not np.isnan([alpha3, beta3]).any() and (alpha3 + beta3) != 0 else np.nan
        else:
            # Not enough data to compute KDE; calculate SCI_3 using all data
            mode3_values = data_nonzero
            mean3 = np.mean(mode3_values) if len(mode3_values) > 0 else np.nan
            variance3 = np.var(mode3_values, ddof=0) if len(mode3_values) > 0 else np.nan
            if variance3 == 0 or mean3 in [0, 1] or np.isnan(mean3):
                alpha3 = beta3 = np.nan
            else:
                alpha3 = mean3 * ((mean3 * (1 - mean3) / variance3) - 1)
                beta3 = (1 - mean3) * ((mean3 * (1 - mean3) / variance3) - 1)
            SCI_3 = (alpha3 - beta3) / (alpha3 + beta3) if not np.isnan([alpha3, beta3]).any() and (alpha3 + beta3) != 0 else np.nan

        # Append the results for each team
        results.append({
            'Team': team,
            'NumMembers': num_unique_members,
            'SCI_1': SCI_1,
            'SCI_2': SCI_2,
            'SCI_3': SCI_3,
            'ValleyPosition': valley_position
        })

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)

    # Fill NaN values in SCI_2 with values from SCI_3
    results_df['SCI_2'] = results_df['SCI_2'].combine_first(results_df['SCI_3'])

    # Optionally, you can drop the SCI_3 column if it's no longer needed
    results_df = results_df.drop(columns=['SCI_3'])

    return results_df
