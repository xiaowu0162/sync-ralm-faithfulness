# sync-ralm-faithfulness
Official Repository for [Synchronous Faithfulness Monitoring for Trustworthy Retrieval-Augmented Generation](https://arxiv.org/abs/2406.13692) (EMNLP 2024 Main)

<span style="color:red">**Work in progress. We expect the data and code release to be done by 2024/10/15.**</span>


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
* The `sentence_level` data is the benchmarking data for context faithfulness tracking mentioned in the paper. It contains prompts, contexts, and the model's output splitted into sentences and attached with the context faithfulness labels. The labels are calculated either by converting the human annotations from RAGTruth or through an NLI model. For further details, refer to Section 4.1 in the paper. 
* The `instance_level` data only contains the prompt and context and is used only for decoding testing.
* Note that the data here includes `famous-100` and `famous-100-anti-v2`, the two new datasets we construct.

We also release the model outputs under the folder `data/rag_outputs`. These outputs will be used for the offline evaluation of SynCheck.

To prepare the data, run the following commands
```
cd data/instance_level ; tar -xzvf * 
cd ragtruth/task_model_split ; python *py
cd ../train_test_split ; python *py
cd ../../../rag_outputs ; tar -xzvf * 
cd ../sentence_level ; tar -xzvf * ; cd ../..
```

#### Setup Environment


#### Install AlignScore

Please follow [this instruction](https://github.com/yuh-zha/AlignScore?tab=readme-ov-file#installation) to install AlignScore from source. Our evaluation additionally requires [downloading the AlignScore-base model checkpoint](https://github.com/yuh-zha/AlignScore?tab=readme-ov-file#checkpoints).

## Offline Feature Calculation

To reproduce the SynCheck results, follow the steps below to calculate three sets of features offline and run training/testing. Note that FOD uses online SynCheck and only requires the offline activation features for LID (see the `FOD and Baselines` section below)


#### Feature 1: Activation

Follow these steps to dump the activations of the last token of each sentence to the disk:

* `cd offline_feature_calc`
* `bash save_sent_last_tok_activation.sh task model split`. 
    * `task` could be `QA`, `Summary`, `Data2txt`, `bio`, `famous-100`, `famous-100-anti-v2`.
    * `model` could be `llama-2-7b-chat` or `mistral-7B-instruct`. 
    * `split` could be `train` or `test` for RAGTruth tasks. For the biology tasks, `split` is always ignored because they only have test splits. 


#### Feature 2: Likelihood

Follow these steps to dump the step-wise distributions to the disk:
* `cd offline_feature_calc`
* `bash save_dist_w_wo_ctx.sh task model mode`. 
    * `task` could be `QA`, `Summary`, `Data2txt`, `bio`, `famous-100`, or `famous-100-anti-v2`.
    * `model` could be `llama-2-7b-chat` or `mistral-7B-instruct`. 
    * `mode` could be `no-rag-cxt` or `rag-cxt`. 


#### Feature 3: AlignScore

Follow these steps to dump the AlignScore for each sentence to the disk:

* `cd backtrack_detection`
* `bash run_detection_alignscore.sh task model split`. 
    * `task` could be `QA`, `Summary`, `Data2txt`, `bio`, `famous-100`, `famous-100-anti-v2`.
    * `model` could be `llama-2-7b-chat` or `mistral-7B-instruct`. 
    * `split` could be `train` or `test` for RAGTruth tasks. For the biology tasks, `split` always defaults to `test`. 

## SynCheck and Baselines

#### Training and Evaluating SynCheck 

To train and evaluate SynCheck offline, you need to calculate three types of offline features in the previous section. Then, follow the two steps below:
* Aggregate features
    * `cd backtrack_detection`
    * `bash aggregate_features.sh task model split`. 
        * `task` could be `QA`, `Summary`, `Data2txt`, `bio`, `famous-100`, `famous-100-anti-v2`.
        * `model` could be `llama-2-7b-chat` or `mistral-7B-instruct`. 
        * `split` could be `train` or `test` for RAGTruth tasks. For the biology tasks, `split` always defaults to `test`. 
* Run training and eval
    * `cd backtrack_detection`
    * `python3 run_classification_agg_features.py --task task --train_task train_task --checked_model model --root_dir root_dir`. 
        * `task` could be `QA`, `Summary`, `Data2txt`, `bio`, `famous-100`, `famous-100-anti-v2`.
        * `train_task` should be same as `task` unless you are experimenting on cross-task faithfulness classification. 
        * `model` could be `llama-2-7b-chat` or `mistral-7B-instruct`. 
        * `root_dir` should be the absolute path of the `sync-ralm-faithfulness` folder. 

In addition, we provide the pre-trained SynCheck checkpoints under `syncheck_checkpoints`.

#### Other Baselines 

We also provide the implementation for the other baselines that you may run. We place all the baselines in `backtrack_detection`. To run any baseline, simply run `bash run_detection_[baseline].sh` and then run `python print_eval_results.py` to print out the scores from the log.

## FOD and Baselines

#### Running FOD 

#### Running CAD and other baselines 

#### Faithfulness-Informativeness Evaluation
