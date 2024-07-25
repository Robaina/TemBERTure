<div align="center">   
<img title="logo" alt="" src="logo.png"  width="600" height="300" align="center">      

<br/><br/>
[![DOI:10.1093/bioadv/vbae103](http://img.shields.io/badge/DOI-10.1093/bioadv/vbae103-F28C28.svg)](https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae103/7713394) 
</div>

We  developed **TemBERTure**, a deep-learning package for protein thermostability prediction based on amino acids sequences. It consists of three components: 

(i) **TemBERTureDB**, a large curated database of thermophilic and non-thermophilic sequences;  
(ii) **TemBERTureCLS**, a classifier  which predicts  the thermal class (non-thermophilic or thermophilic) of a protein sequence;    
(iii) **TemBERTureTm**, a regression model, which predicts the melting temperature of a protein, based on its primary sequence.     

Both models are built upon the existing protBERT-BFD language model [1] and fine-tuned through an adapter-based approach [2], [3]. 

This repository provides implementations and weights for both tasks, allowing users to leverage these models for various protein-related predictive tasks. 

🌟 **Join us on [Discussions](https://github.com/ibmm-unibe-ch/TemBERTure/discussions)!** 🌟

📚 **Have a look at our blog post [here](https://ibmm-unibe-ch.github.io/TemBERTure/)!** 📚

## How to use TemBERTure

#### 1. Download
```
git clone https://github.com/ibmm-unibe-ch/TemBERTure.git
cd TemBERTure
git filter-branch --subdirectory-filter temBERTure -- --all
```
#### 2. Install the python env (python 3.9.18)

**Conda**:
`conda install --file requirements.txt`   
**pip**:
`pip install -r requirements.txt` 

#### 3. Apply TemBERTure on your protein sequences
i.e.: 
```
seq = 'MEKVYGLIGFPVEHSLSPLMHNDAFARLGIPARYHLFSVEPGQVGAAIAGVRALGIAGVNVTIPHKLAVIPFLDEVDEHARRIGAVNTIINNDGRLIGFNTDGPGYVQALEEEMNITLDGKRILVIGAGGGARGIYFSLLSTAAERIDMANRTVEKAERLVREGEGGRSAYFSLAEAETRLDEYDIIINTTSVGMHPRVEVQPLSLERLRPGVIVSNIIYNPLETKWLKEAKARGARVQNGVGMLVYQGALAFEKWTGQWPDVNRMKQLVIEALRR'
```
##### TemBERTureCLS:
```
# Initialize TemBERTureCLS model with specified parameters
from temBERTure import TemBERTure
model = TemBERTure(
    adapter_path='./temBERTure/temBERTure_CLS/',  # Path to the model adapter weights
    device='cuda',                                # Device to run the model on
    batch_size=1,                                 # Batch size for inference
    task='classification'                         # Task type (e.g., classification for TemBERTureCLS)
)
```

```
In [1]: model.predict([seq])
100%|██████████████████████████| 1/1 [00:00<00:00, 22.27it/s]
Predicted thermal class: Thermophilic
Thermophilicity prediction score: 0.999098474215349
Out[1]: ['Thermophilic', 0.999098474215349]
```
##### TemBERTureTM:
```
from temBERTure import TemBERTure

# Initialize all TemBERTureTM replicas with specified inference parameters
model_replica1 = TemBERTure(
    adapter_path='./temBERTure/temBERTure_TM/replica1/',  # Path to the adapter for replica 1
    device='cuda',                                        # Device to run the model on
    batch_size=16,                                        # Batch size for inference
    task='regression'                                     # Task type (e.g., regression for TemBERTureTM)
)

model_replica2 = TemBERTure(
    adapter_path='./temBERTure/temBERTure_TM/replica2/',  # Path to the adapter for replica 2
    device='cuda',                                        # Device to run the model on
    batch_size=16,                                        # Batch size for inference
    task='regression'                                     # Task type (e.g., regression for TemBERTureTM)
)

model_replica3 = TemBERTure(
    adapter_path='./temBERTure/temBERTure_TM/replica3/',  # Path to the adapter for replica 3
    device='cuda',                                        # Device to run the model on
    batch_size=16,                                        # Batch size for inference
    task='regression'                                     # Task type (e.g., regression for TemBERTureTM)
)

```

##### Data Availability: TemBERTureDB

The `/data` folder in this repository contains the sequences used to generate the different datasets for the project. Furthermore, TemBERTureDB can be found on [Zenodo](https://doi.org/10.5281/zenodo.10931927), which also hosts the protein sequences.

### Citing
If you use TemBERTure, please cite the [work](https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae103/7713394?login=false).
```
@article{10.1093/bioadv/vbae103,
    author = {Rodella, Chiara and Lazaridi, Symela and Lemmin, Thomas},
    title = "{TemBERTure: advancing protein thermostability prediction with deep learning and attention mechanisms}",
    journal = {Bioinformatics Advances},
    volume = {4},
    number = {1},
    pages = {vbae103},
    year = {2024},
    month = {07},
    abstract = "{TemBERTure model and the data are available at: https://github.com/ibmm-unibe-ch/TemBERTure.}",
    issn = {2635-0041},
    doi = {10.1093/bioadv/vbae103},
    url = {https://doi.org/10.1093/bioadv/vbae103},
    eprint = {https://academic.oup.com/bioinformaticsadvances/article-pdf/4/1/vbae103/58610069/vbae103.pdf},
}

```

### References
[1] A. Elnaggar et al., “ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 10, pp. 7112–7127, Oct. 2022, doi: 10.1109/TPAMI.2021.3095381.  
[2]	N. Houlsby et al., “Parameter-Efficient Transfer Learning for NLP.” arXiv, Jun. 13, 2019. Accessed: Feb. 14, 2024. [Online]. Available: http://arxiv.org/abs/1902.00751  
[3]	C. Poth et al., “Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning,” 2023, doi: 10.48550/ARXIV.2311.11077.

Thanks to Noah Henrik Kleinschmidt for the TemBERTure logo design.

