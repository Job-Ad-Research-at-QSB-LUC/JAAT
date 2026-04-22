<div align="center">
  <img src="https://github.com/Job-Ad-Research-at-QSB-LUC/JAAT/blob/main/public/logo.png?raw=true" alt="JAAT" width="200"/>

  # Job Ad Analysis Toolkit (JAAT)

  [![PyPI version](https://img.shields.io/pypi/v/jaat.svg)](https://pypi.org/project/jaat/)
  [![PyPI - Downloads](https://img.shields.io/pypi/dm/jaat)]()
  [![GitHub stars](https://img.shields.io/github/stars/Job-Ad-Research-at-QSB-LUC/JAAT.svg?style=social)](https://github.com/Job-Ad-Research-at-QSB-LUC/JAAT/stargazers)
  [![License](https://img.shields.io/github/license/Job-Ad-Research-at-QSB-LUC/JAAT.svg)](https://github.com/Job-Ad-Research-at-QSB-LUC/JAAT/blob/main/LICENSE)

</div>

In this repository, you will find the code for efficient and accurate analysis of job ads. Running the code is simple!

---

## Installation
To install JAAT, simply run the following:

`pip install JAAT`

Then, import JAAT using the following command:

`import JAAT`

or alternatively...

`from JAAT import [MODULE]`

## Utilities and Pipeline Best Practices
The `utils` module of `JAAT` provides helpful tools for managing hardware resources and optimizing throughput for large datasets.

### Hardware Diagnostics and Setup
Before running `JAAT`, use `setup` to verify environment integrity and `diagnostic` to ensure your GPU is correctly recognized.

```python
import JAAT

# Download and verify necessary resources
JAAT.setup()

# Print a diagnostic report 
JAAT.diagnostic()
```

### High-Volume Processing
To process (potentially) millions of text records without crashing your system, use `chunker` to feed modules, `validate_inputs` to ensure correct text format, and `clear_cache` to manage memory between stages.

```python
from JAAT import TaskMatch, utils
utils.toggle_progress(False) # supress progress bars for all modules (they are enabled by default)

TM = TaskMatch()
job_ads = [...] # List of (many) documents / strings

for batch in utils.chunker(job_ads, size=2048):
    valid = validate_inputs(batch)
    results = TM.get_tasks_batch(valid)
    
    # ... save or process your outputs here ...
    
# Clear memory and free up resources before starting the next JAAT module
utils.clear_cache()
```

### Resource Management
For long-running pipelines or scripts that use multiple JAAT modules, use `shutdown` to completely release GPU resources.

```python
JAAT.shutdown()
```

---

## Primary Modules
Below, you will find helpful usage tips for the primary `JAAT` modules. Starting up is quick and easy!

### TaskMatch (v2)
The first module consists of a tool to extract relevant tasks, according to O*NET, given input job ad texts.

After importing the module, simply instantiate the `TaskMatch` object:

```python
TM = JAAT.TaskMatch()
```

Optionally, we can provide a threshold value (default = 0.9, [0, 1]), which governs how lenient to be with the matching (lower means more matches, but potentially less correct ones).

```python
TM = TaskMatch(threshold=0.85)
```

Then, run it on any given (job ad) text:

```python
tasks = TM.get_tasks(TEXT)
```

The output will be a list of tuples, matching an ONET task ID and its title (description). In case no tasks are matched, an empty list will be returned. Example output:

```python
[('16363', 'Identify operational requirements for new systems to inform selection of technological solutions.'), ('16987', 'Prepare documentation or presentations, including charts, photos, or graphs.'), ('9583', 'Assign duties to other staff and give instructions regarding work methods and routines.')]
```

For batch processing, run:

```python
tasks = TM.get_tasks_batch(LIST_OF_TEXTS)
```

### TitleMatch (v2)
The second module assists in matching input job ad titles to coded job titles from O*NET, along with providing useful information about the title.

After importing the module, simply instantiate the `TitleMatch` object:

```python
TM = JAAT.TitleMatch()
```

Then, run it on any given (job ad) text:

```
matched_title = TM.get_title(TEXT)
```

Note that this function will work on either single texts or a list of input texts. The return type will be a list of tuples, each tuple of format:

```python
(MATCHED_TITLE, MATCHED_TITLE_CODE, MATCH_SCORE, TITLE_VALUE, TITLE_FEATURES)
```

Each tuple returned corresponds in order to the input text(s). Note that the returned features will be a semi-colon separated string of feature codes. 

### FirmExtract
The third module is capable of extracting the firm (company) name from a text (not necessarily only job ad texts).

After importing the module, simply instantiate the `FirmExtract` object:

```python
FE = JAAT.FirmExtract()
```

This initiates the firm extraction object with our custom NER model: firmNER. Optionally, you can choose to have all extracted firm names standardized according to the method proposed by [Wasi and Flaeen](https://www.aaronflaaen.com/uploads/3/1/2/4/31243277/wasi_flaaen_statarecordlinkageutilities_20140401.pdf). This can be done by setting the `standardize` parameter to `True`.

Following this, run it on any given (job ad) text:

```python
firms = FE.get_firm(TEXT)
```

This will return a firm name if found, otherwise `None`.

`FirmExtract` also features batch processing. For batch processing, run:

```python
firm_names = FE.get_firm_batch(LIST_OF_TEXTS)
```

This will return a list of firm names (or `None` where no name is found).

### WageExtract
`WageExtract` is used to extract wages (min and max) from a text, as well as the frequency associated with these values (i.e., hourly, weekly, monthly, or annually).

To get started, create a new `WageExtract` object:

```python
W = JAAT.WageExtract()
```

Then, you can classify any text by calling the following function:

```python
prediction = W.get_wage(TEXT)
```

This will return either a dictionary of the extra min/max/frequency values, or the statement `The provided text does not contain a wage statement.`. For larger batches of texts, use the batch function:

```python
predictions = W.get_wage_batch(LIST_OF_TEXTS)
```

This now will return a list predictions in the same order as the inputted texts, with each value either as a dictionary or `None`.

### SkillMatch
`SkillMatch` is a tool to identify and extract required skills from a job posting, very similar to `TaskMatch`. The output of this tool will be all identified skill labels in a provided text, mapped to their respective EuropaCode.

After importing the module, simply instantiate the `SkillMatch` object:

```python
SM = JAAT.SkillMatch()
```

Optionally, we can provide a threshold value (default = 0.87, [0, 1]), which governs how lenient to be with the matching (lower means more matches, but potentially less correct ones).

```python
SM = SkillMatch(threshold=0.8)
```

Then, run it on any given (job ad) text:

```python
skills = SM.get_skills(TEXT)
```

The output will be a list of tuples, matching an skills label to its EuropaCode. In case no skills are matched, an empty list will be returned. Example output:

```python
[('designing systems and products', 'S1.11'), ('work conceptually', 'T2.4'), ('interact with users to gather requirements', 'S1.7')]
```

For batch processing, run:

```python
skills = SM.get_skills_batch(LIST_OF_TEXTS)
```

### AIMatch (beta!)
This module extracts and codifies all AI-related tasks, skills, expertise, and requirements that are stated in job ad texts.

As per usual, load in the module:

```python
AI = JAAT.AIMatch()
```

Following this, `AIMatch` can be used on single texts or in batch mode:

```python
res = AI.get_ai(TEXT)
# or
res = AI.get_ai_batch(LIST_OF_TEXTS)
```

The return value for `get_ai` is a 5-tuple, with the following structure:

```python
(LIST OF MATCHED CONCEPTS/CODES, AVERAGE AI SCORE, TOTAL MATCHES, EXTRACTION CONFIDENCES, MATCH CONFIDENCES)
```

The matched concept/codes are presented in tuples. The "average AI score" is a high-level indicator of the "AI-ness" of the matches, averaged by the number of matches. Both confidence scores are semicolon-delimiter, and they correspond directly to the list of matched concepts. In the case of batch mode, all of these returned values are placed in lists, corresponding to each text input.

## Other Tools

### CREAM (experimental)
`CREAM` is a tool that allows you to extract concepts that are hidden within texts. These concepts, called *classes*, can be defined arbitrarily by you - anything goes! All you need to do is two things:

- **keywords**: each class should contain a list of relevant keywords, or words/phrases that would "trigger" a potential class candidate
- **rules**: *rules* define archetypical text chunks that either support or refute an instance of a defined class given a found keyword. Rules should be manually define using domain expertise, and an arbitrary number of rules may be used.

Keywords should be presented as a list of strings, e.g., `[k1, k2, ..., kn]`.

Rules should be presented as list of tuples, in the form: `[(rule_1, label), (rule_2, label), ..., (rule_n, label)]`.
In the most basic form, the *labels* are binary: 1 denotes the presence of a class, 0 not.

Given these two inputs, one can instantiate the `CREAM` object.

```python
C = JAAT.CREAM(keywords=KEYWORDS, rules=RULES)
```

There are also three optional parameters:

- `class_name`: the name of the class (i.e., labels)
- `n`: useful for CREAM internals, essentially how any context words should be considered on either side of identified keywords
- `threshold`: useful for embedding functions - the minimum similarity threshold a candidate text chunk should meet in order to be matched with a label. The higher the threshold, the stricter the matching criterion.

With this set up, all you need to do is run `CREAM` on a list of texts, and the output will be a DataFrame will the relevant results.

```python
res = C.run(LIST_OF_TEXTS)
```

Specifically, the output will be a Pandas DataFrame will the following columns:

- **text**: the input texts
- **inferred_rule**: the best matching rule, if any
- **inferred_label**: the label assigned based on the best matching rule, if any
- **inferred_confidence**: the "confidence score" of the matching, if a match was made. Note that this is embedding model specific and should be interpreted relatively.

### JobTag
`JobTag` is used to classify pieces of texts (such as job ads) according to expert defined classification schemes. This is done using niche classifiers which we also release publicly here.

As of now the following classes are supported:
`['CitizenshipReq', 'GovContract', 'VisaExclude', 'VisaInclude', 'WorkAuthReq', 'driverslicense', 'ind_contractor', 'proflicenses', 'wfh', 'yesunion']`

To get started, create a new `JobTag` object by doing the following:

```python
J = JAAT.JobTag(class_name=CLASS)
```

where `CLASS` is replaced by one of the supported classes. Optionally, you can also specify an `n` parameter (default: 4), which defines how large of a context window around keywords to consider.

Then, you can classify any text (binary classification, 1 == positive) by calling the following function:

```python
prediction = J.get_tag(TEXT)
```

This will return a tuple of the form `(class_name, 1/0)`. For larger batches of texts, use the batch function:

```python
predictions = J.get_tag_batch(LIST_OF_TEXTS)
```

This now will return a list of 1/0 predictions, in the same order as the input texts.

### Readability
This module provides a simple utility for calculating the Flesch-Kincaid readability score for a job posting text or texts. To start:

```python
R = JAAT.Readability()
```

Then, simply run on a text or batch of texts:

```python
score = R.get_readability(TEXT)
# or
scores = R.get_readability_batch(LIST_OF_TEXTS)
```

The returned scores are floats, rounded to two decimal places.

### ConceptSearch
This module is a simple and highly efficient tool to extract "concepts" from a corpus of texts. These concepts are defined by keywords, and they can be mapped to any arbitrary amount of rule values.

In the `automatons` directory, we provide some pre-packaged concept maps, which can be used directly in `ConceptSearch`. Additionally, you can create your own, with the following structure:

```json
CONCEPTS = {
    "keyword1": ("value1", "value2", ...),
    "keyword2": ("value1", "value2", ...),
    "keyword3": ("value1", "value2", ...),
    ...
}
```

Using this structure, you can initialize `ConceptSearch` in one of the following two ways:

```python
CS = JAAT.ConceptSearch(concept_map=CONCEPTS)
# or
CS = JAAT.ConceptSearch(concept_file=/path/to/concepts.pkl)
```

Following this, `ConceptSearch` can be used on single texts or in batch mode:

```python
res = CS.get_concepts(TEXT)
# or
res = CS.get_concepts_batch(LIST_OF_TEXTS)
```

The returned objects will be a list of tuples for each text, wherein the tuples represent the value tuples of the found (matched) keywords.

## Acknowledgements

This project has received generous support from the National Labor Exchange, the Russell Sage Foundation, the Washington Center for Equitable Growth.

### Citation
If you find `JAAT` useful in your research, please consider citing our working paper that introduces many of the abovementioned modules:

```
@article{meisenbacher2025extracting,
  title={Extracting O* NET Features from the NLx Corpus to Build Public Use Aggregate Labor Market Data},
  author={Meisenbacher, Stephen and Nestorov, Svetlozar and Norlander, Peter},
  journal={arXiv preprint arXiv:2510.01470},
  year={2025}
}
```
