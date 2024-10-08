# sync-ralm-faithfulness
Official Repository for [Synchronous Faithfulness Monitoring for Trustworthy Retrieval-Augmented Generation](https://arxiv.org/abs/2406.13692) (EMNLP 2024 Main)

**Work in progress. We expect the data and code release to be done by 2024/10/15.**


## Directory Structure

```
sync-ralm-faithfulness/
├── data/                     # All required data
├── offline_feature_calc/     # Code to calculate the required features
├── backtrack_detection/      # SynCheck and baselines
├── decoding/                 # FOD and baselines
├── syncheck_checkpoints/     # SynCheck checkpoints for various tasks
├── LICENSE
├── README.md
├── requirements.txt
```

## Setup

#### Download Data

The data for SynCheck and FOD evaluation is placed in `data/sentence_level` and `data/instance_level`, respectively. 

The `sentence_level` data is the benchmarking data for context faithfulness tracking mentioned in the paper. It contains prompts, contexts, and the model's output splitted into sentences and attached with the context faithfulness labels. The labels are calculated either by converting the human annotations from RAGTruth or through an NLI model. For further details, refer to Section 4.1 in the paper. 

The `instance_level` data only contains the prompt and context and is used only for decoding testing.

#### Setup Environment


#### Install AlignScore


## Offline Feature Calculation

To reproduce the SynCheck results, follow the steps below to calculate three sets of features offline and run training/testing. Note that FOD uses online SynCheck and only requires the offline activation features for LID (see the `FOD and Baselines` section below)


#### Feature 1: Activation


#### Feature 2: Likelihood


#### Feature 3: AlignScore



## SynCheck and Baselines

#### Training and Evaluating SynCheck 

#### Other Baselines 

## FOD and Baselines

#### Running FOD 

#### Running CAD and other baselines 

#### Faithfulness-Informativeness Evaluation
