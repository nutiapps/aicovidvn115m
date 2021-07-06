![](https://img.shields.io/github/license/nutiapps/aicovidvn115m?style=plastic)
# AICovidVN115m - 2nd place solution of warm up phase

We build very simple and efficient models using trees and various sets of acoustic features. Due to small data and noise we aim to run multiple trials then average results to get the robust scores. 

### Problem

Please go to this site: https://www.covid.aihub.vn/

### Usage

The experiment dag is

```
                +----------+                
                | download |                
                +----------+                
                      *                     
                      *                     
                      *                     
             +---------------+              
             | featurization |              
             +---------------+              
              **            **              
            **                **            
          **                    **          
+--------------+            +------------+  
| trainpredict |            | evaluation |  
+--------------+            +------------+  

```
Please follow these steps to reproduce:

1. Install required packages/libs using `pip install -r requirements.txt`
2. Set params `trainpredict.trials` to 20 in `params.yaml` file.
3. Activate to new venv `source venv/bin/active`
4. Run command `dvc repro` and go to data/subs to get the zip file.
5. Show experiment metric `dvc metrics show`

```
Path              avg_prec    roc_auc                                 
data/scores.json  0.79949     0.89643
```
6. Show experiment `dvc exp show --include-params evaluation.model_index,evaluation.collection_index`

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Experiment              ┃ Created  ┃ avg_prec ┃ roc_auc ┃ evaluation.model_index ┃ evaluation.collection_index ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ workspace               │ -        │ 0.771    │ 0.87558 │ 1                      │ 2                           │
│ master                  │ 08:49 AM │ 0.771    │ 0.87558 │ 1                      │ 3                           │
│ ├── a17ea25 [exp-982cd] │ 10:13 AM │ 0.81382  │ 0.90145 │ 0                      │ 5                           │
│ ├── 7f21409 [exp-8d70b] │ 10:12 AM │ 0.80894  │ 0.90005 │ 0                      │ 3                           │
│ ├── f8cb8e6 [exp-33324] │ 09:53 AM │ 0.77155  │ 0.88196 │ 0                      │ 2                           │
│ ├── ed9e7f9 [exp-82a35] │ 09:39 AM │ 0.79949  │ 0.89643 │ 1                      │ 5                           │
│ ├── 158f5a9 [exp-0e5bd] │ 09:39 AM │ 0.79305  │ 0.89234 │ 1                      │ 3                           │
│ └── 2fef00f [exp-b4d92] │ 09:33 AM │ 0.771    │ 0.87558 │ 1                      │ 2                           │
└─────────────────────────┴──────────┴──────────┴─────────┴────────────────────────┴─────────────────────────────┘

```

7. fine tuning with `dvc exp run -S evaluation.model_index=1 -S evaluation.collection_index=2`

8. run mutiplt exps at same time 
- `dvc exp run --queue -S evaluation.model_index=0 -S evaluation.collection_index=2`
- `dvc exp run --queue -S evaluation.model_index=0 -S evaluation.collection_index=3`
- `dvc exp run --queue -S evaluation.model_index=0 -S evaluation.collection_index=5`
- `dvc exp run --run-all --jobs 3`

**NOTE**
Evaluation is used to check single score on cross validation folds of train dataset. It's not used for making prediction. This is the end-to-end reproducible pipeline to make a submission on private test.

### Extra information

1. Total training and making predictions is about 2 hours.
- feature extraction: `825.07 seconds`
- single model evaluation: `380.57 seconds`

2. To update and add the resnet-based for the next phase.

3. To add references of related works/papers/competitions.

### References

1. ![Yamnet](https://tfhub.dev/google/yamnet/1)
2. ![Opensmile](https://audeering.github.io/opensmile/about.html)

### License
MIT