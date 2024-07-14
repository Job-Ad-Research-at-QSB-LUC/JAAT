# Job Ad Analysis Toolkit (JAAT)
In this repository, you will find the code for efficient and accurate analysis of job ads. Running the code is simple!

## Getting Started
### TaskMatch
The first module consists of a tool to extract relevant tasks, according to O*NET, given input job ad texts.

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

### TitleMatch
The second module assists in matching input job ad titles to coded job titles from O*NET.

After importing the module, simply instantiate the `TitleMatch` object:

`TM = TitleMatch()`


Then, run it on any given (job ad) text:

`matched_title = TM.get_title(TEXT)`

Note that this function will work on either single texts or a list of input texts. The return type will be a list of tuples, each tuple of format:

`(MATCHED_TITLE, MATCHED_TITLE_CODE, MATCH_SCORE)`

Each tuple returned corresponds in order to the input text(s).

### FirmExtract
The third module is capable of extracting the firm (company) name from a text (not necessarily only job ad texts).

After importing the module, simply instantiate the `FirmExtract` object:

`FE = FirmExtract()`

This initiates the firm extraction object with our custom NER model: firmNER. Following this, run it on any given (job ad) text:

`firms = FE.get_firm(TEXT)`

This will return a firm name if found, otherwise `None`.

`FirmExtract` also features batch processing. For batch processing, run:

`firm_names = FE.get_firm_batch(LIST_OF_TEXTS)`

This will return a list of firm names (or `None` where no name is found).