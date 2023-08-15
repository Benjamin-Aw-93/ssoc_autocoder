# SSOC Autocoder

> **Quick links below:**  
> See [Clickup](https://app.clickup.com/3825139/v/l/s/43633588) for sprint  
> See [Google Drive](https://drive.google.com/drive/u/1/folders/1a4zV5ILczikQnraScjXRwi2XCNEQtfwA) for documentation and architecture  
> See [AWS](https://project-finch.signin.aws.amazon.com/console) for datasets and deployment (account alias is `project-finch`)

## General Info

* Project started on 1 Aug 2021
* Team: Shaun Khoo, Benjamin Aw

## Timeline

Phase 2: Deployment (ongoing)
* Apr 2022: Discussion and preparation
* May 2022: ?
* Jun 2022: ?
* Jul 2022: ?
* Aug 2022: Anniversary review

<s>Phase 1: Development (completed)
* Aug 2021: Data processing
* Sep 2021: Data processing
* Oct 2021: Initial modelling + training on AWS
* Nov 2021: Data augmentation and manual tagging
* Dec 2021: Language modelling + model training
* Jan 2022: Deployment of POC
* Mar 2022: Presentation to Data Steering Committee</s>

## Style Guide

* General
  * Notebooks are for exploration and development, scripts are for production. If you need to iterate quickly or you're doing a one-off job, then use notebooks.
  * Always comment your code! 
* Notebooks
  * Follow this naming convention: `{YYYY-MM-DD} - {name of notebook}`. This helps to track when the notebook was created and keeps things organised.
  * Turn the first cell of the notebook into Markdown, and follow the convention below. Leave 2 spaces at the end of each line to do a 'line break'.
    * `## {name of notebook}`
    * `**Author:** {your name}`
    * `**Date:** {DD MMM YYYY}`
    * `**Context:** {why are you doing this, or any other relevant details to help people understand}`
    * `**Objective:** {what you hope to do/achieve from this notebook}`
  * Use different notebooks for different things - try not to mix them together unless you are just exploring / iterating quickly.
  * Okay to start with a messy notebook, but clean it up with generous Markdown documentation and comments in the code once you are done.
* Scripts
  * Scripts should be marked as an importable package `ssoc-autocoder` with the appropriate `__init__.py` files. See [Python documentation](https://docs.python.org/3/tutorial/modules.html#packages)
  * Functions for similar purposes can be placed in the same script file (eg `data-processing` or `modelling`)  
  * Docstrings should follow the [Google style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) - PyCharm will render this correctly for you
  * Each function should have at least one unit test, even if they are only called by other functions.
  * Mark integration tests as separate so we can run them separately. See [StackOverflow](https://stackoverflow.com/questions/54898578/how-to-keep-unit-tests-and-integrations-tests-separate-in-pytest)
* Git
  * Raise an issue for any production code issue so it can be documented and addressed properly
  * Do not push `Data` and `Models` folders into the repository. Keep only code in the repo.  
  * Committing
    * Commit frequently (while you're on a branch - see below), even if it's a draft
    * Read [these general guidelines](https://chris.beams.io/posts/git-commit/) for Git commit messages
    * Always have a subject line that uses the imperative
    * Always include a body message to explain the changes and context (doesn't have to be very long)  
  * Pull requests  
    * Create a PR for any non-trivial code changes. PRs should be created in response to issues. Only hotfixes (small, urgent) are excluded from this rule
    * Name the PR intuitively. Use "feature-XXX_XXX_XXX". For example, "feature-add-unit-tests"
    * PRs must be approved by the other party, not by yourself. 
    * When merging PRs, always use squash and merge  

## Parking Lot

* Interpretability and fairness
  * Understand how and why the model is predicting specific SSOCs
  * Fairness assessment of the types of JDs and the respective error rates
 

## Technical Documentations

### 1) Key changes made to model architecture and parameters in July 2023
  * Learning rate reduced from 0.005 to 0.001
  * Language model component seperated from hidden layers
    * Embeddings done seperately before training
    * Embeddings are stored in S3 bucket
    * Embeddings is retrieved from S3 bucket during training
   * Model accuracy improved from 49.9% to >80%
   * Duration for one epoch reduced from 4 hours/epoch to 10 minutes/epoch

### 2) Experiments done between July 2023 - August 2023
  * Using ChatGPT-3 generated job ads
  * Tested using XGBoost
  * Ran model training with more epochs

|          Model         	|   Details  	|   Epochs   	|     1D ACC    	|     2D ACC    	|     3D ACC    	|     4D ACC    	|     5D ACC    	|     Top-3     	|     Top-5     	|
|:----------------------:	|:----------:	|:----------:	|:-------------:	|:-------------:	|:-------------:	|:-------------:	|:-------------:	|:-------------:	|:-------------:	|
| Hierachical Classifier 	|  W/O GenAI 	|      35    	|     92.70%    	|     89.79%    	|     87.13%    	|     84.29%    	|     79.38%    	|     91.52%    	|     94.39%    	|
| Hierachical Classifier 	|  W/O GenAI 	|     100    	|     94.01%    	|     91.90%    	|     89.55%    	|     87.23%    	|     84.01%    	|     94.01%    	|     96.47%    	|
| Hierachical Classifier 	| With GenAI 	|     100    	|     93.91%    	|     92.15%    	|     90.42%    	|     88.44%    	|     84.50%    	|     93.70%    	|     96.23%    	|
|  XGBoost max_depth = 3 	|  W/O GenAI 	|      -     	|     89.03%    	|     85.88%    	|     83.46%    	|     82.04%    	|     79.69%    	|     91.38%    	|     93.88%    	|
| XGBoost max_depth = 60 	|  W/O GenAI 	|      -     	|     90.93%    	|     88.62%    	|     86.51%    	|     84.57%    	|     82.32%    	|     92.49%    	|     94.78%    	|
 
## Model Architecture Diagram
<img width="710" alt="image" src="https://github.com/Benjamin-Aw-93/ssoc_autocoder/assets/66168700/4f58f3d2-6841-41d1-8e91-05666d461995">


