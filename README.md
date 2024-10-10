# sync-ralm-faithfulness
Official Repository for [Synchronous Faithfulness Monitoring for Trustworthy Retrieval-Augmented Generation](https://arxiv.org/abs/2406.13692) (EMNLP 2024 Main)

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
cd ../all_split ; python *py
cd ../../../rag_outputs ; tar -xzvf * 
cd ../sentence_level ; tar -xzvf * ; cd ../..
```

#### Setup Environment

We recommend using a conda environment for the project. You may follow the steps below to set up.
```
conda create -n syncheck python=3.9
conda activate syncheck
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
We have tested this environment on a Linux machine with CUDA 12.1. If you use a different platform, you may need to modify the requirements.

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

#### Running FOD and other baselines

FOD calculates the features on-the-fly during decoding, feeds the features into a pre-trained SynCheck checkpoint, and leverages SynCheck's outputs to guide the direction of decoding. The only offline feature is the activations on the train sets, which is required to compute LID. To run FOD, 
* Make sure you compute the feature 1 in the offline features.
* `cd decoding`
* `bash run_fod.sh task model beam_size sample_size_per_round temperature start_beam_search_syncheck_threshold stop_beam_threshold`
    * `task` could be `QA`, `Summary`, `Data2txt`, `bio`, `famous-100`, `famous-100-anti-v2`.
    * `model` could be `llama-2-7b-chat` or `mistral-7B-instruct`. 
    * `beam_size` is the K in the paper. We used 2 for our experiments.
    * `sample_size_per_round` is the S in the paper. We used 6 for our experiments.
    * `temperature` is for proposing the next sentence continuation. We used 0.7 in the paper.
    * `start_beam_search_syncheck_threshold` is the threshold on SynCheck's scores when backtrack is triggered. We used 0.8 in the paper.
    * `stop_beam_threshold` is the threshold for pruning out example proposals. We used 0.7 in the paper.

We also provide the implementation for CAD, the major baseline we compared with in the paper. To run CAD, use the command `bash run_cad.sh task model cad_alpha`.

#### Faithfulness-Informativeness Evaluation

To evaluate the outputs from the decoding algorithms, follow these steps:
* Install FActScore by following the instructions [here](https://github.com/shmsw25/FActScore).
* `cd decoding/evaluation`
* `bash eval.sh task pred_file`

The decoding script will decompose the outputs into propositions and compare each proposition against the retrieved context using the llama+npm method proposed in the FActScore paper. Finally, the script will print out the fact-level accuracy (faithfulness) and the number of decomposed atomic facts (informativeness).


## Citation

If you find the work useful, please cite:

```
@article{wu2024syncheck,
      title={Synchronous Faithfulness Monitoring for Trustworthy Retrieval-Augmented Generation}, 
      author={Di Wu and Jia-Chen Gu and Fan Yin and Nanyun Peng and Kai-Wei Chang},
      year={2024},
      eprint={2406.13692},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.13692}, 
}
```
