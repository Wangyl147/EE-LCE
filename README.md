# EE-LCE

- This repo releases our implementation for the EE-LCE model.
- It uses GPT-3.5 to enhance event extaction datasets.
- It is built based on the pretrained Flan T5 model, and finetuned on our data.

## Requirements

Our main experiments and analysis are conducted on the following environment:

- CUDA (11.4)
- cuDNN (10.1.243)
- Pytorch (1.11.0)
- Transformers (4.4.2)
- datasets (1.17.0)
- nltk (3.6.5)

You can install the required libraries by running 

```
pip install -r requirements.txt
```


## Data

Our models are trained and evaluated on **IE INSTRUCTIONS**. 
You can find the explained data in the **IE INSTRUCTIONS/** directory.


## Training

Script for training the EE-LCE model in our paper can be found at [`scripts/train_flan-t5.sh`](scripts/train_flan-t5.sh). You can run it as follows:

```
bash ./scripts/train_flan-t5.sh
```




## Evaluation

Script for evaluating the EE-LCE model in our paper can be found at [`scripts/eval_flan-t5.sh`](scripts/eval_flan-t5.sh). You can run it as follows:

```
bash ./scripts/eval_flan-t5.sh
```
The decoded results would save to predict_eval_predictions.jsonl in your output dir. 

F1 can be calculated by running **src/calculate_f1.py**.
```
python src/calculate_f1.py
```

## Citation

<!-- ```latex@article{}``` -->





