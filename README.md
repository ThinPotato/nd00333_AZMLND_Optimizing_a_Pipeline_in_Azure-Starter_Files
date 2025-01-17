# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
This dataset contains data about different jobs and the financial, educational, and marital situations of the people working them. We seek to predict the y column representing yes and no.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best performing model was an AutoML model using a VotingEnsemble algorithm with an accuracy score of 0.91736.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The pipeline follows 3 major steps: cleaning data, tuning the hyperparameter and running the experiment. 
Firstly, we clean the data. In this case, this mostly means turning strings into numbers so the computer can process. For example, Yes is turned into 1, no into 0, feburary into 2, etc...
Next we config the hyperparameter experiment. I ended up using RandomParameterSampling with a C value of any number between 0.05 and 0.1 and a max_iter value of a random choice of 16, 32, 64, and 128. These configurations allow for our hyperparameter to have a variety of approaches and attempts when tuning.
Finally, we run the experiment with a primary goal of maximizing accuracy. After all is said and done, this gives us a respectable best case accuracy score of 0.9115.

**What are the benefits of the parameter sampler you chose?**

As I learned in the previous lessons, the RandomParameterSampling sampler offers the best compromise of accuracy and efficiency. It's a solid choice that doesn't lose out on much accuracy, but saves a ton of time processing. 

**What are the benefits of the early stopping policy you chose?**
Like the Random Parameter Sampling, the Bandit Policy is a solid choice for ending the experiment when results are good enough that continuing running any longer would be a waste.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

AutoML generated a model with a higher accuracy score (0.9173) using the VotingEnsemble algorithm. Its most valuable features were: Duration, nr.employed, cons.conf.idx, etc...

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

Yes there were differences. I had to wrangle the data a tad more to get the autoML to work. Additionally, the autoML took longer to process. I was seeing about double the time to finish the experiment. At the end, the autoML managed to pull ahead with a slightly better accuracy score, but we'd have to see how it performs on a larger scale to be able to decide if its worth the longer processing time. Any differences in output can probably be attributed to autoML simply trying more approaches using different algorithms; remember, it had double the time to think.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
I think it would be beneficial to gather a larger dataset. Perhaps more columns in similar nature to duration, as that was the most valuable feature. Additionally, while using autoML did give us a slightly higher score, overall, it's probably not worth the time. It does make a good example as a demonstration of its weaknesses, but in terms of experimentation, I'd leave it in favor of a manually selected approach. If you wanted to look at this question in the opposite light, giving the autoML even more time to process would likely give us a higher model score, but again, probably not worth the time overall.

## Proof of cluster clean up
**Image of cluster marked for deletion**
I cannot screenshot it, but I have deleted my compute instance and Compute cluster under the Compute tab in the Manage section of the ML Studio. Check notebook for removal code...
