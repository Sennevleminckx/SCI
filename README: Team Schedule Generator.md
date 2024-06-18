# README: Team Schedule Generator

This repository contains Python code designed to generate a collaborative schedule for teams based on specified details and constraints. The generated schedule includes shifts assigned to collaborators over a defined period.

## Dependencies

	• pandas (version >= 1.0.0): For data manipulation and analysis.
	• numpy (version >= 1.18.0): For numerical computing.
	• datetime: For date and time manipulation.

Ensure these packages are installed before running the script.

## Team Details

The team_details dictionary specifies:

	• start_date: Start date for the schedule.
	• end_date: End date for the schedule.
	• teams: This is a nested dictionary under the teams key, where each key represents a unique team identifier (team_bk). 
    	• team_size_fulltime: Number of full-time collaborators in the team.
    	• team_size_parttime: Number of part-time collaborators in the team. (= random work % of between 0.5 - 0.9)
    	• shifts: List of strings representing the shifts available for the team. Each string format is "HHMM-HHMM", indicating the start and end times of a shift.
    	• max_shifts_per_collaborator: Maximum number of shifts that can be assigned to a single collaborator.
    	• freedom_level: Optional parameter indicating the flexibility level for shift assignment. Default is 1.0, where 1.0 means full flexibility. (note: very senstive parameter, increments of 	0.001 suffice)

## Workflow

	1.	Date Range Initialization: Generates a list of all dates within the specified start_date and end_date.
	2.	Adjusting Maximum Days: Calculates the maximum days each collaborator can work based on their proportion (full-time or part-time).
	3.	Shuffling Dates: Randomizes the order of dates to introduce variability in scheduling.
	4.	Generating Collaborators:
  		• Iterates over each team and creates collaborators based on team size.
  		•Assigns shifts and work dates to each collaborator while considering their full-time or part-time status and the specified maximum shifts.
	5.	Creating Schedule Entries: Constructs schedule entries with collaborator IDs, team IDs, start times, and end times for each assigned shift.
	6.	Output: Returns a DataFrame (schedule_df) containing the generated schedule entries, sorted by collaborator and start time.

## Usage

To generate a schedule:

	• Define team_details with the required parameters.
	• Execute the script containing the provided code snippet.

## Notes

	• The script ensures randomness in shift assignments and flexibility in scheduling, respecting team-specific constraints.
	• Adjustments such as shift preferences and freedom levels can be configured in team_details to tailor the scheduling process.
