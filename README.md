# AICovidVN115m - 2nd place solution of warm up phase

We build very simple and efficient models using trees and various sets of acoustic features. Due to small data and noise we aim to run multiple trials then average results to get the robust scores. 

### Steps

1. To install required packages/libs using `pip install -r requirements.txt`
2. To reproduce full experiment simple adjust params `trials` to 20 and call `dvc repro`.

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

3. total training and making predictions is about 2 hours.
- feature extraction: `825.07 seconds`
- single model evaluation: `380.57 seconds`

### License
MIT