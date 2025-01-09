def calculate_SCI_scores(df):
    """
    Calculate Structured Collaboration Index (SCI) scores for each team in the dataframe,
    considering overlaps only when collaborators are working on the same task.

    Parameters:
    - df (DataFrame): The input dataframe containing at least the following columns:
        'start', 'end', 'Team', 'collaborator_bk', 'Task'

    Returns:
    - results_df (DataFrame): A dataframe containing SCI scores and additional information for each team.
    """
    import warnings
    import numpy as np
    import pandas as pd
    from itertools import combinations
    from tqdm import tqdm
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import brentq

    # Suppress specific FutureWarning
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="use_inf_as_na option is deprecated and will be removed in a future version"
    )

    def calculate_overlap(df1, df2):
        """
        Calculate the total overlap in hours between two collaborators, considering tasks.
        """
        total_overlap = 0.0

        # Find the set of tasks that both collaborators have worked on
        tasks_shared = set(df1['Task']).intersection(set(df2['Task']))

        for task in tasks_shared:
            df1_task = df1[df1['Task'] == task]
            df2_task = df2[df2['Task'] == task]

            start1 = df1_task['start'].values.astype('datetime64[ns]')
            end1 = df1_task['end'].values.astype('datetime64[ns]')
            start2 = df2_task['start'].values.astype('datetime64[ns]')
            end2 = df2_task['end'].values.astype('datetime64[ns]')

            latest_start = np.maximum(start1[:, None], start2)
            earliest_end = np.minimum(end1[:, None], end2)
            overlap = (earliest_end - latest_start) / np.timedelta64(1, 'h')
            overlap[overlap < 0] = 0

            total_overlap += np.sum(overlap)

        return total_overlap

    def compute_SCI(mode_values):
        """
        Compute the SCI value for a set of mode values.
        """
        if len(mode_values) == 0:
            return np.nan

        mean = np.mean(mode_values)
        variance = np.var(mode_values, ddof=0)

        # If no variance or mean is 0 or 1, SCI is not defined
        if variance == 0 or mean in [0, 1] or np.isnan(mean):
            return np.nan

        alpha = mean * ((mean * (1 - mean) / variance) - 1)
        beta = (1 - mean) * ((mean * (1 - mean) / variance) - 1)

        if np.isnan(alpha) or np.isnan(beta) or (alpha + beta) == 0:
            return np.nan

        SCI = (alpha - beta) / (alpha + beta)
        return SCI

    def find_gaussian_intersection(gmm, x_range):
        """
        Find intersection point between two Gaussian components in a GMM.
        """
        def gaussians_diff(x):
            # Weighted log probability difference of the two components
            return (
                    gmm.weights_[0] * gmm._estimate_weighted_log_prob(np.array([[x]]))[:, 0] -
                    gmm.weights_[1] * gmm._estimate_weighted_log_prob(np.array([[x]]))[:, 1]
            )[0]

        try:
            intersection = brentq(gaussians_diff, x_range[0], x_range[1])
            return intersection
        except ValueError:
            return None

    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    results = []

    for team in tqdm(df['Team'].unique(), desc='Processing Teams', unit='team'):
        team_data = df[df['Team'] == team]
        collaborators = team_data['collaborator_bk'].unique()
        num_unique_members = len(collaborators)

        total_hours_per_collaborator = team_data.groupby('collaborator_bk').apply(
            lambda x: ((x['end'] - x['start']).sum().total_seconds() / 3600)
        )

        matrix = pd.DataFrame(0.0, index=collaborators, columns=collaborators)

        # Calculate overlaps for all pairs
        for collaborator1, collaborator2 in combinations(collaborators, 2):
            df1 = team_data[team_data['collaborator_bk'] == collaborator1]
            df2 = team_data[team_data['collaborator_bk'] == collaborator2]

            total_overlap = calculate_overlap(df1, df2)
            total_hours_collaborator1 = total_hours_per_collaborator[collaborator1]
            total_hours_collaborator2 = total_hours_per_collaborator[collaborator2]

            normalized_overlap_1 = total_overlap / total_hours_collaborator1 if total_hours_collaborator1 > 0 else 0
            normalized_overlap_2 = total_overlap / total_hours_collaborator2 if total_hours_collaborator2 > 0 else 0

            matrix.loc[collaborator1, collaborator2] = normalized_overlap_1
            matrix.loc[collaborator2, collaborator1] = normalized_overlap_2

        stacked_matrix = matrix.stack()
        data_off_diagonal = stacked_matrix[
            stacked_matrix.index.get_level_values(0) != stacked_matrix.index.get_level_values(1)
            ]

        data_values = data_off_diagonal.values
        data_nonzero = data_values[data_values > 0]

        SCI_team = np.nan
        SCI_ext = np.nan
        SCI_core = np.nan
        valley_position = None

        if len(data_nonzero) > 0:
            # If there's no variance or all values are identical, skip GMM
            unique_values = np.unique(data_nonzero)
            if len(unique_values) == 1:
                # All values are the same
                SCI_team = compute_SCI(data_nonzero)
                # Only one mode, so no separation
                SCI_ext = np.nan
                SCI_core = np.nan
                valley_position = unique_values[0]  # All points are the same
            else:
                SCI_team = compute_SCI(data_nonzero)

                data_nonzero_reshaped = data_nonzero.reshape(-1, 1)
                gmm2 = GaussianMixture(n_components=2, random_state=42)

                # Try fitting the GMM
                try:
                    gmm2.fit(data_nonzero_reshaped)
                except ValueError:
                    # If fitting fails, fallback to mean threshold
                    threshold = data_nonzero.mean()
                    valley_position = threshold
                    mode1_values = data_nonzero[data_nonzero < threshold]
                    mode2_values = data_nonzero[data_nonzero >= threshold]
                    SCI_ext = compute_SCI(mode1_values)
                    SCI_core = compute_SCI(mode2_values)
                    results.append({
                        'Team': team,
                        'NumMembers': num_unique_members,
                        'SCI_team': SCI_team,
                        'SCI_ext': SCI_ext,
                        'SCI_core': SCI_core,
                        'ValleyPosition': valley_position,
                    })
                    continue

                x_min, x_max = data_nonzero.min(), data_nonzero.max()
                weights = gmm2.weights_

                # Check if effectively one cluster
                if np.isclose(weights[0], 1.0) or np.isclose(weights[1], 1.0):
                    # Use mean as threshold
                    threshold = data_nonzero.mean()
                else:
                    intersection = find_gaussian_intersection(gmm2, (x_min, x_max))
                    if intersection is not None and x_min < intersection < x_max:
                        threshold = intersection
                    else:
                        threshold = data_nonzero.mean()

                valley_position = threshold
                mode1_values = data_nonzero[data_nonzero < threshold]
                mode2_values = data_nonzero[data_nonzero >= threshold]

                SCI_ext = compute_SCI(mode1_values)
                SCI_core = compute_SCI(mode2_values)

        results.append({
            'Team': team,
            'NumMembers': num_unique_members,
            'SCI_team': SCI_team,
            'SCI_ext': SCI_ext,
            'SCI_core': SCI_core,
            'ValleyPosition': valley_position,
        })

    results_df = pd.DataFrame(results)
    return results_df
