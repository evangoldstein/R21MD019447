#!/bin/bash

source /activate/your/environment
# List of label columns to use
labels=("romanticrelationshipissues" "romanticrelationshipendbreaku" "feelingdepressedmelancholystr" "druguse" "argumentwithromanticpartner" "alcoholuse" "drugalcoholusehistory" "knownsuspectedinfidelity" "generalmedicalconditionknown" "legalissues" "reachingoutforhelpthreatenin" "anotherpersonsfirearmaccessi" "psychiatricmentalhealthhospit" "multiplefirearms" "mentalhealthconditionknown" "medication" "deathoffamilyfriendromantic" "historyoffamilyfriendsuicid" "foundincar" "foundoutsideofresidence" "domesticviolence" "foundbyfamilyromanticpartner" "foundatresidence" "foundbylawenforcement" "historyofsuicidalideations")

# Loop over each label and run the Python script
counter=1
for label in "${labels[@]}"
do
  echo "Running for label: $label with ID: $counter"
  python training.py --label_col "$label" --label_id "$counter"
  ((counter++))
done