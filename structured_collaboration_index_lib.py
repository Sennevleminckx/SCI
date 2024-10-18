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

    Parameters:
    - df1 (DataFrame): DataFrame for collaborator 1 with 'start' and 'end' columns.
    - df2 (DataFrame): DataFrame for collaborator 2 with 'start' and 'end' columns.

    Returns:
    - total_overlap (float): The total overlap in hours.
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
    Compute the SCI value for a set of mode values.

    Parameters:
    - mode_values (array-like): The mode values to compute SCI for.

    Returns:
    - SCI (float): The computed SCI value.
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
    Find the intersection point between two Gaussian components in a GMM.

    Parameters:
    - gmm (GaussianMixture): The fitted Gaussian Mixture Model with two components.
    - x_range (tuple): The range of x values to search for the intersection.

    Returns:
    - intersection (float): The x-value where the two Gaussians intersect.
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

    Parameters:
    - df (DataFrame): The input dataframe containing at least the following columns:
        'start', 'end', 'Team', 'collaborator_bk'

    Returns:
    - results_df (DataFrame): A dataframe containing SCI scores and additional information for each team.
    """
    # Ensure time columns are in datetime format
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['Team'] = df['Team'].astype(int)

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

            # Normalize the overlap
            max_collab_hours = min(
                total_hours_per_collaborator[collaborator1],
                total_hours_per_collaborator[collaborator2]
            )
            normalized_overlap = total_overlap / max_collab_hours if max_collab_hours > 0 else 0

            # Update the matrix
            matrix.loc[collaborator1, collaborator2] = normalized_overlap
            matrix.loc[collaborator2, collaborator1] = normalized_overlap

        # Extract the upper triangle of the matrix, excluding the diagonal
        upper_triangle_indices = np.triu_indices_from(matrix, k=1)
        upper_triangle_values = matrix.values[upper_triangle_indices]

        # Filter out zeros to focus on actual overlaps
        data_nonzero = upper_triangle_values[upper_triangle_values > 0]

        # Initialize variables
        SCI_team = np.nan
        SCI_ext = np.nan
        SCI_core = np.nan
        valley_position = None
        is_bimodal = False

        # Proceed if we have data
        if len(data_nonzero) > 0:
            # Compute SCI_team
            SCI_team = compute_SCI(data_nonzero)

            # Proceed to check bimodality if enough data
            if len(data_nonzero) > 1:
                data_nonzero_reshaped = data_nonzero.reshape(-1, 1)

                # Fit GMMs with 1 and 2 components
                gmm1 = GaussianMixture(n_components=1)
                gmm2 = GaussianMixture(n_components=2)

                gmm1.fit(data_nonzero_reshaped)
                gmm2.fit(data_nonzero_reshaped)

                # Compute BIC scores
                bic1 = gmm1.bic(data_nonzero_reshaped)
                bic2 = gmm2.bic(data_nonzero_reshaped)

                # Select the model with the lower BIC
                if bic2 < bic1:
                    # Bimodal distribution
                    is_bimodal = True

                    # Find intersection point between the two Gaussians
                    x_min, x_max = data_nonzero.min(), data_nonzero.max()
                    x_range = (x_min, x_max)
                    intersection = find_gaussian_intersection(gmm2, x_range)

                    if intersection is not None and x_min < intersection < x_max:
                        threshold = intersection
                        valley_position = threshold

                        # Separate data into mode 1 and mode 2 based on the threshold
                        mode1_values = data_nonzero[data_nonzero < threshold]
                        mode2_values = data_nonzero[data_nonzero >= threshold]

                        # Calculate SCI_ext for values below the valley (mode 1)
                        SCI_ext = compute_SCI(mode1_values)

                        # Calculate SCI_core for values above the valley (mode 2)
                        SCI_core = compute_SCI(mode2_values)
                    else:
                        # Unable to find a valid intersection
                        pass  # SCI_ext and SCI_core remain np.nan
                else:
                    # Unimodal distribution
                    pass  # SCI_ext and SCI_core remain np.nan
            else:
                # Not enough data to check bimodality
                pass  # SCI_ext and SCI_core remain np.nan
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
            'IsBimodal': is_bimodal
        })

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df
