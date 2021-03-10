# COCON_ICLR2021
This is our Pytorch implementation of COCON. 

**CoCon: A Self-Supervised Approach for Controlled Text Generation (ICLR 2021)**<br>
*Alvin Chan, Yew-Soon Ong, Bill Pung, Aston Zhang, Jie Fu*<br>
https://arxiv.org/abs/2010.02684

TL;DR: We propose CoCon to control the content of text generation from LMs by conditioning on content inputs at an interleave layer.


## Requirements
- Python 3.7.6 on Linux
- PyTorch 1.4

## Dependencies
Install dependencies with: 
```bash
pip install -r requirements.txt
```

## Dataset
1. Download COCON's training data from https://github.com/openai/gpt-2-output-dataset
2. Place the `medium-345M-k40.${split}.jsonl` files inside the `data/gpt2output/` folder


## COCON Training
Train COCON with a GPT-2 language model, with the parameters reported in the paper:  
```bash
sh train_cocon.sh
```  
After training, the COCON block's weights will be saved as `models/COCON/cocon_block_pytorch_model.bin`.

### Training Key Arguments
`--do_train` : whether to train COCON or not
`--output_dir` : directory of COCON weights  
`--model_name_or_path` : type of language model to train COCON with
`--output_hidden_for_cocon_after_block_ind` : index of transformer block whose hidden states are used as input to COCON for content conditioning, value is 6 for results reported in paper, meaning that the output of GPT-2's 7th transformer block is used as COCON block's input.


### Pretrained COCON weights
You can download COCON's pretrained weights [here](https://drive.google.com/file/d/10bZrIxfQY7xDDqgrfrhbN_zgj7SfQKIL/view?usp=sharing) and save it in `models/COCON/` to start generating with COCON.


## COCON Controlled Generation
Sample script on how to generate COCON sentiment-controlled text:  
```bash
sh generation/generate_cocon_sentiments.sh
```  

Sample script on how to generate COCON topic-controlled text:  
```bash
sh generation/generate_cocon_topics.sh
```

### Generation Key Arguments
`--do_cocon_compute` : whether to do COCON generation
`--output_dir` : directory of COCON block's weights  
`--model_name_or_path` : type of language model
`--cocon_output_filename` : path of saved generation samples  
`--cocon_compute_history_source_data_file` : filename of text file containing prompt texts for generation  
`--cocon_compute_context_source_data_file` : filename of text file containing target content for generation


## Summary of Key Folders/Files
- `transformers/`: code for models and optimizers
- `transformers/modeling_gpt2.py`: code for COCON block and GPT-2 language model
- `BOW/`: target content tokens used for COCON topic control
- `attr_markers/`: target content tokens used for COCON sentiment control
- `prompts/`: prompt text used for text generation


## Citation
If you find our repository useful, please consider citing our paper:

```
@inproceedings{
chan2021cocon,
title={CoCon: A Self-Supervised Approach for Controlled Text Generation},
author={Alvin Chan and Yew-Soon Ong and Bill Pung and Aston Zhang and Jie Fu},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=VD_ozqvBy4W}
}
```


## Acknowledgements
Code is based largely on:
- https://github.com/huggingface/transformers