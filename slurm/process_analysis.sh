#! /bin/bash

if [ "$1" != "" ]; then
  JOB_ID=$1
  echo "Job ID: $JOB_ID"
else 
  echo "Wrong command line arguments"
fi

if [ "$2" != "" ]; then
  TASK_ID=$2
  echo "Task ID: $TASK_ID"
else
  echo "Wrong command line arguments"
fi

# Define output path from relevant sub-path of input file
OUTPUT_DIR="/rstorage/james/qhat/$JOB_ID"
mkdir -p $OUTPUT_DIR

# Run python script via pipenv
cd /software/users/james/jetscape-docker/STAT
pipenv run python run_all_models.py -c analysis_config.yaml -o $OUTPUT_DIR -i $(($TASK_ID-2))

# Move stdout to appropriate folder
mv /rstorage/james/qhat/slurm-${JOB_ID}_${TASK_ID}.out /rstorage/james/qhat/${JOB_ID}/
