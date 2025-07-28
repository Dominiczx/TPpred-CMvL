# TPpred-CMvL

This repository scripts for running the analysis described in the paper **TPpred-CMvL: Prediction of Multi-Functional Therapeutic Peptide Using Contrast Multi-View Learning**.

<img width="1272" height="871" alt="Framework-of-TPpred-CMvL" src="https://github.com/user-attachments/assets/ea2f7d80-3e6d-4848-bb49-2386eccfcc69" />



## Setup/Installation

1. Clone this repository

```
git clone https://github.com/Dominiczx/TPpred-CMvL.git
```

2. Install and Activate Conda Environment with Required Dependencies

```
conda env create -f environment.txt
conda activate TPpred-CMvL
```

3. Navigate to repository

```
cd TPpred-CMvL
```

## Model Training

To train this model on training dataset, run

```bash
python tape_model.py
```

Additional arguments:

```
--print_freq: print frequency
--save_freq: save frequency
--batch_size: batch_size
--num_workers: num of workers to use
--epochs: number of training epochs
--learning_rate: learning_rate
--lr_decay_epochs: where to decay lr, can be a list
--lr_decay_rate: decay rate for learning rate
--weight_decay: weight_decay
--pretrained_model: pretrained model path
--model: type of the model
--p: parameter for random mutation
--temp: temperature for loss function
--cosine: using cosine annealing
--syncBN: using synchronized batch normalization
--warm: warm-up for large batch training
--trial: id for recording multiple runs
--pssm_weight: The Weight of pssm feature
```

NCE Loss parameters:

```
--nce_k: K
--nce_t: T
--nce_m: M
--feat_dim: dim of feat for inner product
--softmax: using softmax contrastive loss rather than NCE
```

All the arguments above has its default parameter. You can also modify them in tape_model.py

## Visualize Classification 

If you want to compare the feature result before and after classification, run:

```
python visual.py
```

It could generate the picture similar to Fig.3 in **TPpred-CMvL: Prediction of Multi-Functional Therapeutic Peptide Using Contrast Multi-View Learning**



If you visit [zenodo.org](https://zenodo.org/uploads/16354699),  create a "TPpred-CMvL" folder and extract "data.rar" and "code.rar" to this folder. Enter this folder and run the above code.
