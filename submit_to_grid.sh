#!/bin/bash

# Script to submit the GMC grid jobs

source /exp/seaquest/app/software/osg/users/chleung/SL7/Distribution/setup/setup.sh
source /cvmfs/seaquest.opensciencegrid.org/seaquest/software/SL7/seaquest/kTrackerRun5/setup.sh
export PYTHONPATH=/pnfs/e906/persistent/users/mhossain/seaquest_scripts/pylib:$PYTHONPATH

./runGMC.py --grid --preset=run3 --Record=ROOT --server=e906-db4.fnal.gov --raw-name=scratch_20_Feb_LH2 --n-events=50000 --n-subruns=5 --Target=H --EventPosition=Target --Generator=DY --Acceptance=Acc --grid-args="--expected-lifetime=24h" --first-subrun=1
