# ChordSeqAI: Generating Chord Sequences Using Deep Learning
## 1. Introduction
Creating chord sequences for a song is one of the first obstacles musicians face when composing music. Ensuring that the final progression is both interesting and fitting into the genre can be a difficult task for inexperienced composers, as merely understanding the space of possible chords does not guarantee a satisfactory result. Developing an open-source AI-powered tool for chord sequence generation could simplify the start of a music career for many artists. While some solutions already try to solve this problem, they are often limited in chord diversity, support for multiple genres, or not being open-source. In this project, I want to focus on these problems, so in the sections below, I describe my vision of how the process may be unveiled.

## 2. Establishing a goal
Instead of completely replacing the musician, the tool would enhance the creative potential of each user by enabling them to take charge of the musical direction. Ideally, I imagine a web application for artists to generate chord sequences. Upon choosing the first chord of the sequence (given a distribution based on the genre of the music), the musician will see a landscape of following chords with their appropriate probability. By choosing a chord from the suggestions, the model would again predict the next chord, until the musician is satisfied with the resulting sequence. Then the output could be converted into a desired format for the artist to download and use freely in their piece.

## 3. Project scope
### 3.1. Data gathering and processing
The model can only be as good as the training data is, thus creating a robust dataset is critical to constructing any model. A collection of some publicly available datasets mixed with data scraped from music banks can be used to create a diverse dataset. While chord progressions are not subject to copyright, it is still important to be responsible about which data is used.

### 3.2. Exploratory data analysis
Before training the model on the available data, it would be helpful to know what is the dataset composed of. I will allocate time to exploring the dataset and seeing which interesting facts can be obtained from it, such as the most commonly used keys and chord types for each of the genres.

### 3.3. Context-based embeddings
For more complex chords, one-hot encoding may not be enough to train an effective model, therefore context-based embeddings, inspired by the GloVe method, could be used before training the model on the dataset.

### 3.4. Model architecture
I will explore several different architectures, starting from recurrent neural networks using GRUs and LSTMs, then move on to using the Transformer architecture. For style-based generation, I may try to input the style as another token, or perhaps use an adversarial approach with controllable generation, replacing the LayerNorm with AdaIN (adaptive instance normalization), inspired by the StyleGAN paper. The architectures tried and the final one being used will be determined by experiments, where I'll iteratively try new approaches.

### 3.5. Training
For experimentation and training, I plan to use Kaggle, as it offers free 30 NVIDIA Tesla P100 GPU hours per week, which should be enough for this task. I intend to use PyTorch for model implementation and training due to its simplicity and popularity.

### 3.6. Evaluation
Evaluating the performance of generative models can be challenging. Typically, the goal is to compare the data distributions between the real and generated examples. Since chord progressions are not as complex as image generation, some metrics could be applied directly to the outputs (without using feature extraction). I think I will also try n-gram analysis, perplexity, and some rule-based evaluation inspired by music theory. To obtain the final performance, I will experiment with human evaluation, seeing whether real and generated chord sequences can be distinguished and how would people rate each progression based on several criteria.

### 3.7. Deployment
The model will be converted into the ONNX format for faster inference and then it may be deployed serverless using Azure Functions. Subsequently, I aim to create a custom API for communication with Azure.

### 3.8. Web application
To make the model more accessible to broader audiences, I plan to create a Next.js application equipped with a simple user interface. This website will then be deployed on Azure Static Web Apps.

## 4. Timeline
Given the complexity of the project, the four-month timeframe is intensive. The project has to be completed by the end of this year, so I propose this schedule:
* September: Dataset creation, exploratory data analysis, initial model development
* October: Iterative development
* November: Evaluation, deployment, web application
* December: Documentation, time buffer for unforeseen challenges

## 5. Conclusion
In this project I aim to develop a powerful open-source tool for generating chord sequences, effectively democratizing the early stages of music production. While I have set a timeline and a series of milestones I want to achieve, some unexpected challenges may arise, which could alter the direction of the project. Anyway, I hope that this will be a successful project that makes the world a better place.