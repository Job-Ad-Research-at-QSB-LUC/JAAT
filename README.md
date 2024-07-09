# TaskMatch
In this repository, you will find the code for efficient and accurate task labeling of job ads. Running the code is simple!

## Getting Started
After importing the module, simply instantiate the `TaskMatch` object:

`TM = TaskMatch()`

Optionally, we can provide a threshold value (default = 0.9, [0, 1]), which governs how lenient to be with the matching (lower means more matches, but potentially less correct ones).

`TM = TaskMatch(threshold=0.85)`

Then, run it on any given (job ad) text:

`tasks = TM.get_tasks(TEXT)`

The output will be a list of tuples, matching an ONET task ID and its title (description). In case no tasks are matched, an empty list will be returned. Example output:

`[('16363', 'Identify operational requirements for new systems to inform selection of technological solutions.'), ('16987', 'Prepare documentation or presentations, including charts, photos, or graphs.'), ('9583', 'Assign duties to other staff and give instructions regarding work methods and routines.')]`

For batch processing, run:

`tasks = TM.get_tasks_batch(LIST_OF_TEXTS)`