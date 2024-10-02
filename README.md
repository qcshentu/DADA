```
./DADA
|-- DADA
|   |-- config.json
|   |-- configuration_DADA.py
|   |-- modeling_DADA.py
|   `-- pytorch_model.bin
|-- data_provider
|   |-- __init__.py
|   |-- batch_scheduler.py
|   |-- data_factory.py
|   |-- data_loader.py
|   `-- read_data.py
|-- dataset
|   `-- evaluation_dataset
|-- exp
|   |-- __init__.py
|   `-- exp_DADA.py
|-- metrics
|   |-- __init__.py
|   |-- affiliation
|   |-- metrics.py
|   `-- spot.py
|-- run.py
`-- scripts
    |-- MSL
    |-- NIPS_CICIDS
    |-- NIPS_Creditcard
    |-- NIPS_GECCO
    |-- NIPS_SWAN
    |-- PSM
    |-- SMAP
    |-- SMD
    |-- SWAT
    |-- affiliation.sh
    `-- auc.sh
```

DADA requires transformers==4.33.3
## Evaluation
- Prepare the benchmark datasets.  
Datasets can be downloaded at this link: https://drive.google.com/file/d/1QumS8bSRsLZT7u5TWLaWctDWvGnSyeRB/view?usp=drive_link  
- Running the following command to evaluate.

```bash
# affiliation metric for all datasets
sh ./scripts/affiliation.sh
# [Example] Evaluate on MSL.
# sh ./scripts/MSL/affiliation.sh
```
```bash
# auc_roc metric for all datasets
sh ./scripts/auc.sh
# [Example] Evaluate on MSL.
# sh ./scripts/MSL/auc.sh
```
