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
    Calculate the total overlap in hours between two collaborators.

    Args:
        df1 (pandas.DataFrame): DataFrame for collaborator 1 with 'start' and 'end' columns.
        df2 (pandas.DataFrame): DataFrame for collaborator 2 with 'start' and 'end' columns.

    Returns:
        float: The total overlap in hours.
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

def compute_SCI(mode_values):
    """
    Compute the Structured Collaboration Index (SCI) value for a set of mode values.

    Args:
        mode_values (array-like): The mode values to compute SCI for.

    Returns:
        float: The computed SCI value, or np.nan if not computable.
    """
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

def find_gaussian_intersection(gmm, x_range):
    """
    Find the intersection point between two Gaussian components in a Gaussian Mixture Model (GMM).

    Args:
        gmm (GaussianMixture): The fitted GMM with two components.
        x_range (tuple): The range of x values to search for the intersection.

    Returns:
        float or None: The x-value where the two Gaussians intersect, or None if not found.
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

def calculate_SCI_scores(df):
    """
    Calculate Structured Collaboration Index (SCI) scores for each team in the dataframe.

    Args:
        df (pandas.DataFrame): The input DataFrame containing at least the following columns:
            - 'start': datetime
            - 'end': datetime
            - 'Team': int
            - 'collaborator_bk': str

    Returns:
        pandas.DataFrame: A DataFrame containing SCI scores and additional information for each team.
    """
    # Ensure time columns are in datetime format
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    # df['Team'] = df['Team'].astype(int)  # Commented out as per your update

    # Initialize list to collect results
    results = []

    # Process each team
    for team in tqdm(df['Team'].unique(), desc='Processing Teams', unit='team'):
        team_data = df[df['Team'] == team]
        collaborators = team_data['collaborator_bk'].unique()
        num_unique_members = len(collaborators)

        # Calculate total hours per collaborator
        total_hours_per_collaborator = team_data.groupby('collaborator_bk').apply(
            lambda x: ((x['end'] - x['start']).sum().total_seconds() / 3600)
        )

        # Initialize the matrix for storing normalized overlap values
        matrix = pd.DataFrame(0.0, index=collaborators, columns=collaborators)

        # Calculate overlaps between all pairs of collaborators
        for collaborator1, collaborator2 in combinations(collaborators, 2):
            df1 = team_data[team_data['collaborator_bk'] == collaborator1]
            df2 = team_data[team_data['collaborator_bk'] == collaborator2]

            # Calculate total overlap between two collaborators
            total_overlap = calculate_overlap(df1, df2)

            # Normalize the overlap from each collaborator's perspective
            total_hours_collaborator1 = total_hours_per_collaborator[collaborator1]
            total_hours_collaborator2 = total_hours_per_collaborator[collaborator2]

            normalized_overlap_1 = total_overlap / total_hours_collaborator1 if total_hours_collaborator1 > 0 else 0
            normalized_overlap_2 = total_overlap / total_hours_collaborator2 if total_hours_collaborator2 > 0 else 0

            # Update the matrix with asymmetric values
            matrix.loc[collaborator1, collaborator2] = normalized_overlap_1
            matrix.loc[collaborator2, collaborator1] = normalized_overlap_2

        # Stack the matrix to get all pairwise overlaps
        stacked_matrix = matrix.stack()

        # Remove self-overlaps (diagonal elements)
        data_off_diagonal = stacked_matrix[stacked_matrix.index.get_level_values(0) != stacked_matrix.index.get_level_values(1)]

        # Get the overlap values
        data_values = data_off_diagonal.values

        # Filter out zeros to focus on actual overlaps
        data_nonzero = data_values[data_values > 0]

        # Initialize variables
        SCI_team = np.nan
        SCI_ext = np.nan
        SCI_core = np.nan
        valley_position = None

        # Proceed if we have data
        if len(data_nonzero) > 0:
            # Compute SCI_team using all data
            SCI_team = compute_SCI(data_nonzero)

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

            # Separate data into mode 1 and mode 2 based on the threshold
            mode1_values = data_nonzero[data_nonzero < threshold]
            mode2_values = data_nonzero[data_nonzero >= threshold]

            # Calculate SCI_ext for values below the threshold (mode 1)
            SCI_ext = compute_SCI(mode1_values)

            # Calculate SCI_core for values above the threshold (mode 2)
            SCI_core = compute_SCI(mode2_values)

        else:
            # No data to compute SCI_team
            pass  # SCI_team remains np.nan

        # Append the results for each team
        results.append({
            'Team': team,
            'NumMembers': num_unique_members,
            'SCI_team': SCI_team,
            'SCI_ext': SCI_ext,
            'SCI_core': SCI_core,
            'ValleyPosition': valley_position,
        })

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df
