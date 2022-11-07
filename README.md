# EMSE-2022

Scripts of our paper submitted to Empirical Software Engineering (EMSE)

### Package Requirement

To run this code, some packages are needed as follows:

```
Python version == 3.6
scikit-learn == 0.24.2
imbalanced-learn == 0.7.0
imbalanced-ensemble == 0.1.4
wittgenstein == 0.1.6
scipy == 1.2.1
```

Here is the directory structure of our repository:

```
├─ datasets  -> the original dataset
├─ cross ->  the data for cross project prediction
├─ whole ->  the data for prediction among the whole dataset
├─ classifiers.py -> the implementation of classification models
├─ data_loader.py -> the implementation of loading dataset as the dataframe
├─ data_processor.py -> the implementation of data preprocessing
├─ evaluation.py -> the implemantation of performance indicators
├─ im_ensembles.py -> the implementation of all ensemble techniques
├─ im_samplings.py -> the implementation of all sampling techniques
├─ im_weights.py -> the implementation of weighted classification models
├─ main.py -> the entry to obtain main prediction results
├─ main_cross.py -> the entry to obtain cross prediction results
├─ main_whole.py -> the entry to obtain prediction results among the whole dataset
├─ main_error.py -> the entry to analysis the errors
├─ results.zip -> the detailed experimental results
├─ results_processor.py -> the implementation of processing experimental results
├─ trainer.py -> the implementation to train the main predictive models
├─ trainer_cross.py -> the implementation to train the cross project models
├─ trainer_error.py -> the implementation to analyze the error data
├─ trainer_whole.py -> the implementation to train the predictive models on the whole dataset
```

### Quickly Start

(1) Create the subdirectory ````results/```` in current directory 

(2) Run ````python main.py```` 

(3) The results will save in the directory ````results/````

(4) Similarly, we can also run ````python main_cross.py```` to obtain the results of the cross-project prediction.


