#!/bin/sh

echo -----------------------------------------------------------------------------------------------------
echo "End to end solution, run steps in sequence, it takes about 2 hours to get the submission"
echo -----------------------------------------------------------------------------------------------------
date

echo "(*) download dataset"
nohup python src/create_dataset.py >> ./log.log 2>&1 &
wait

echo "(*) extract features"
nohup python src/featurization.py >> ./log.log 2>&1 &
wait

echo "(*) train and predict"
nohup python src/train_predict.py >> ./log.log 2>&1 &
wait

date
echo -----------------------------------------------------------------------------------------------------
echo "Please check file in subs/*.zip and submit it."
echo -----------------------------------------------------------------------------------------------------
echo "DONE. HAPPY MODELING <3"
echo "Contact saigonapps@gmail.com for further information."


