# ChordSeqAI: Generating Chord Sequences Using Deep Learning

GitHub repository storing the source code and resources of this graduate project. Contains the scraper and datasets, trained models, and Jupyter notebooks detailing the project's development together with additional files. This repository is a segment of the larger ChordSeqAI project and primarily serves as a resource for those interested in the technical aspects of AI-driven music composition.

## Project Structure

### Data

This section includes the dataset and any additional large files used within the notebooks. Uploaded using [Git LFS](https://git-lfs.com/), if you also want to clone this data, use `git lfs pull`.

### Models

Trained PyTorch models, stored using Git LFS.

### Notebooks

Jupyter notebooks, the main part of this project. Below is outlined the suggested order in which to go through them.

- Exploratory Data Analysis
- Data Tokenization
- Recurrent Network
- Transformer
- Conditional Transformer
- Classification Transformer
- Style Extraction
- Style Transformer
- Sequence Generation
- Model Evaluation

### Report

The LaTeX technical report together with an exported PDF file. To understand this project further, it is recommended to take a look at it.

### src

Source code used throughout the notebooks, parsing the chords and representing sequences of tokens in a readable way. The `models` directory provides a simple way to apply the trained models elsewhere.

### Additional files

- scraper.py: used to obtain the datasets
- Project Vision.md: written before starting this project, describing its goals
- requirements.txt: a file containing all the Python libraries used in the code, install the dependencies by `pip install -r requirements.txt`

## Other resources

The web application can be found in the [ChordSeqAI Web App repository](https://github.com/StudentTraineeCenter/chord-seq-ai-app).
