# AICovidVN115m - 2nd place of warm up phase

Our second place solution at aihub.vn.

### Steps

```
# 1. run download stage
dvc run -n download \              
-d src/create_dataset.py -o data/raw \
python src/create_dataset.py

# 2. run featurization

# 3. run train_predict

# 4. run evaluate to see the local cv5 score
```

### License
MIT