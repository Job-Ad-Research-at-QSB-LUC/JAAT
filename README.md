# Job Ad Analysis Toolkit (JAAT)
In this repository, you will find the code for efficient and accurate analysis of job ads. Running the code is simple!

## Installation
To install JAAT, simply run the following:

`pip install JAAT`

Then, import JAAT using the following command:

`from JAAT import JAAT`

## TaskMatch
The first module consists of a tool to extract relevant tasks, according to O*NET, given input job ad texts.

After importing the module, simply instantiate the `TaskMatch` object:

`TM = JAAT.TaskMatch()`

Optionally, we can provide a threshold value (default = 0.9, [0, 1]), which governs how lenient to be with the matching (lower means more matches, but potentially less correct ones).

`TM = TaskMatch(threshold=0.85)`

Then, run it on any given (job ad) text:

`tasks = TM.get_tasks(TEXT)`

The output will be a list of tuples, matching an ONET task ID and its title (description). In case no tasks are matched, an empty list will be returned. Example output:

`[('16363', 'Identify operational requirements for new systems to inform selection of technological solutions.'), ('16987', 'Prepare documentation or presentations, including charts, photos, or graphs.'), ('9583', 'Assign duties to other staff and give instructions regarding work methods and routines.')]`

For batch processing, run:

`tasks = TM.get_tasks_batch(LIST_OF_TEXTS)`

## TitleMatch
The second module assists in matching input job ad titles to coded job titles from O*NET.

After importing the module, simply instantiate the `TitleMatch` object:

`TM = JAAT.TitleMatch()`

Then, run it on any given (job ad) text:

`matched_title = TM.get_title(TEXT)`

Note that this function will work on either single texts or a list of input texts. The return type will be a list of tuples, each tuple of format:

`(MATCHED_TITLE, MATCHED_TITLE_CODE, MATCH_SCORE)`

Each tuple returned corresponds in order to the input text(s).

## FirmExtract
The third module is capable of extracting the firm (company) name from a text (not necessarily only job ad texts).

After importing the module, simply instantiate the `FirmExtract` object:

`FE = JAAT.FirmExtract()`

This initiates the firm extraction object with our custom NER model: firmNER. Optionally, you can choose to have all extracted firm names standardized according to the method proposed by [Wasi and Flaeen](https://www.aaronflaaen.com/uploads/3/1/2/4/31243277/wasi_flaaen_statarecordlinkageutilities_20140401.pdf). This can be done by setting the `standardize` parameter to `True`.

Following this, run it on any given (job ad) text:

`firms = FE.get_firm(TEXT)`

This will return a firm name if found, otherwise `None`.

`FirmExtract` also features batch processing. For batch processing, run:

`firm_names = FE.get_firm_batch(LIST_OF_TEXTS)`

This will return a list of firm names (or `None` where no name is found).

## CREAM
`CREAM` is a tool that allows you to extract concepts that are hidden within texts. These concepts, called *classes*, can be defined arbitrarily by you - anything goes! All you need to do is two things:

- **keywords**: each class should contain a list of relevant keywords, or words/phrases that would "trigger" a potential class candidate
- **rules**: *rules* define archetypical text chunks that either support or refute an instance of a defined class given a found keyword. Rules should be manually define using domain expertise, and an arbitrary number of rules may be used.

Keywords should be presented as a list of strings, e.g., `[k1, k2, ..., kn]`.

Rules should be presented as list of tuples, in the form: `[(rule_1, label), (rule_2, label), ..., (rule_n, label)]`.
In the most basic form, the *labels* are binary: 1 denotes the presence of a class, 0 not.

Given these two inputs, one can instantiate the `CREAM` object.

`C = JAAT.CREAM(keywords=KEYWORDS, rules=RULES)`

There are also three optional parameters:

- `class_name`: the name of the class (i.e., labels)
- `n`: useful for CREAM internals, essentially how any context words should be considered on either side of identified keywords
- `threshold`: useful for embedding functions - the minimum similarity threshold a candidate text chunk should meet in order to be matched with a label. The higher the threshold, the stricter the matching criterion.

With this set up, all you need to do is run `CREAM` on a list of texts, and the output will be a DataFrame will the relevant results.

`res = C.run(LIST_OF_TEXTS)`

Specifically, the output will be a Pandas DataFrame will the following columns:

- **text**: the input texts
- **inferred_rule**: the best matching rule, if any
- **inferred_label**: the label assigned based on the best matching rule, if any
- **inferred_confidence**: the "confidence score" of the matching, if a match was made. Note that this is embedding model specific and should be interpreted relatively.

## ActivityMatch
In a similar way to `TaskMatch`, `ActivityMatch` will extract general activity statements from your texts, according to a set of predefined daily activities (see `data/lexiconwex2023.csv`).

`AM = JAAT.ActivityMatch()`

Optionally, we can provide a threshold value (default = 0.9, [0, 1]), which governs how lenient to be with the matching (lower means more matches, but potentially less correct ones).

`AM = ActivityMatch(threshold=0.85)`

Then, run it on any given text:

`activities = AM.get_activities(TEXT)`

For batch processing, run:

`activities = AM.get_activities_batch(LIST_OF_TEXTS)`

## JobTag
`JobTag` is used to classify pieces of texts (such as job ads) according to expert defined classification schemes. This is done using niche classifiers which we also release publicly here.

As of now the following classes are supported:
`['CitizenshipReq', 'GovContract', 'VisaExclude', 'VisaInclude', 'WorkAuthReq', 'driverslicense', 'ind_contractor', 'proflicenses', 'wfh', 'yesunion']`

To get started, create a new `JobTag` object by doing the following:

`J = JAAT.JobTag(class_name=CLASS)`

where `CLASS` is replaced by one of the supported classes. Optionally, you can also specify an `n` parameter (default: 4), which defines how large of a context window around keywords to consider.

Then, you can classify any text (binary classification, 1 == positive) by calling the following function:

`prediction = J.get_tag(TEXT)`

This will return a tuple of the form `(class_name, 1/0)`. For larger batches of texts, use the batch function:

`predictions = J.get_tag_batch(LIST_OF_TEXTS)`

This now will return a list of 1/0 predictions, in the same order as the input texts.

## Acknowledgements

This project has received generous support from the National Labor Exchange, the Russell Sage Foundation, the Washington Center for Equitable Growth.

### Data Citation
In the demo notebook `JAATDemo.ipynb` and the companion slides, we use the data made available by the following publication:

```Zhou, Steven, John Aitken, Peter McEachern, and Renee McCauley. “Data from 990 Public Real-World Job Advertisements Organized by O*NET Categories.” Journal of Open Psychology Data 10 (November 21, 2022): 17. https://doi.org/10.5334/jopd.69.```