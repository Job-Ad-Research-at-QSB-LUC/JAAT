## [0.9.0] - 2026-04-17
Our biggest update yet! `JAAT` has been modularized and optimized for ease of use and even faster results.

### Changed
 - All `JAAT` modules can now be imported individually, allowing for faster load times. Simply call `from JAAT import XYZ` with your module of choice.
 - Match-based modules have been optimized for speed.
 - Other minor optimizations and improvements to allow for faster compuation and less RAM load.

### Removed
 - Old data files and demos have been archived.


## [0.8.3] - 2026-04-12
### Changed
 - `JobTag` is now more lightweight. Pre-trained classifiers are lazily downloaded and cached as needed, making the installation experience much faster.
 - `TaskMatch`, `SkillMatch`, and `AIMatch` now feature the ability to customize the underlying embedding and classification models used. Check out our Hugging Face to see the progression of models available to use. By default, these modules use the most recently published classification models. **Note**: if you change the embedding model, you will likely need to toggle the similarity threshold for best results!

### Removed
The compressed ML models for `JobTag` have been removed from the `JAAT` repository and package. They are now hosted on our [Hugging Face page](https://huggingface.co/loyoladatamining/JobTag).

---

## [0.8.1/2] - 2026-04-09
### Fixed
Minor bug fixes in `SkillMatch`.

---

## [0.8.0] - 2026-04-02
### Added
`AIMatch` is now fully functional! We have cleaned up the underlying data files for more accurate coding.

### Changed
`TaskMatch` and `SkillMatch` are now better - both are powered by stronger classification models which have been further fine-tuned in a LLM+human validation round.

### Fixed
`TaskMatch` in the source code (not package) not matches the package version.

### Removed
The old coding scheme for the `AIMatch` beta has been removed. Please take note of the new codes.

---

## Pre [0.8.0]
Version history starts here!