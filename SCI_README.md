# SCI
## Structured Collaboration Index

This repository contains Python code designed to analyze team collaboration data and compute a collaboration index (SCII) for each team based on their collaboration patterns. The analysis involves calculating overlap metrics between collaborators within each team and deriving statistical estimates of collaboration intensity.
### Dependencies
	•	pandas (version >= 1.0.0): For data manipulation and analysis.
	•	numpy (version >= 1.18.0): For numerical computing.
	•	tqdm (version >= 4.0.0): For progress bars during computation.
	•	itertools: For generating combinations of collaborators.

Make sure these packages are installed before running the script.
### Input:
   collaborator_bk Team               start                 end
0           T01_1  T01 2021-02-04 21:01:00 2021-02-05 06:59:00
1           T01_1  T01 2021-01-14 21:01:00 2021-01-15 06:59:00
2           T01_1  T01 2021-01-04 21:01:00 2021-01-05 06:59:00
3           T01_1  T01 2021-01-23 21:01:00 2021-01-24 06:59:00
4           T01_1  T01 2021-01-10 21:01:00 2021-01-11 06:59:00
	•	collaborator_bk: Unique identifier for each collaborator.
	•	Team: Unique identifier for each team.
	•	start: Start date and time of collaboration.
	•	end: End date and time of collaboration.

 The start and end columns are converted to datetime objects using pandas.

### Workflow
	1.	Data Preprocessing: Convert start and end columns to datetime objects.
	2.	Team-wise Analysis:
	  •	Iterate over each unique team in the dataset.
	  •	For each team, calculate total collaboration hours per collaborator.
	  •	Construct an overlap matrix between all pairs of collaborators within the team.
	  •	Normalize overlap values based on the total collaboration hours of the collaborators involved.
	3.	Statistical Analysis:
	  •	Flatten the overlap matrix and filter out zero values.
	  •	Compute sample mean and variance of non-zero overlap values.
	  •	Estimate parameters (alpha and beta) of the Beta distribution using method of moments.
	4.	Output:
	  •	Store results in a DataFrame including Team, number of unique members (NumMembers), estimated Alpha, Beta, and the SCII (SCII) computed from these parameters.
	  •	Save results to a CSV file (SCII_analysis_4.csv) in the ./output/ directory.

## Usage

To run the analysis:

	•	Ensure all dependencies are installed.
	•	Execute the script containing the provided code snippet.

## Notes

	•	The SCII (SCII) provides insights into team collaboration intensity, with higher values indicating stronger collaboration patterns.
	•	Ensure the input data is correctly formatted with datetime columns for accurate analysis.

