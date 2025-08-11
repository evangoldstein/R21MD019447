#!/bin/bash

source /activate/your/environment

# List of label columns to use
labels=("romanticrelationshipissues" "romanticrelationshipendbreaku" "feelingdepressedmelancholystr" "argumentwithromanticpartner" "druguse" "alcoholuse" "knownsuspectedinfidelity" "legalissues" "medication" "deathoffamilyfriendromantic" "anotherpersonsfirearmaccessi" "psychiatricmentalhealthhospit" "multiplefirearms")

# Loop over each label and run the Python script
counter=1
for label in "${labels[@]}"
do
  echo "Running for label: $label with ID: $counter"
  python training_k_cv_seeds.py --label_col "$label" --label_id "$counter"
  ((counter++))
done