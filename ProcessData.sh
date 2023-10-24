#!/bin/bash

# echo "===> make simple tree"
# python SimpleTree.py
#
# echo "===> split data"
# python SplitTree.py
#
# echo "===> make unet data"
# root -b -q MakeUNetData.cc


while getopts ":stump" opt;
do
    case ${opt} in
        s ) echo "===> make simple tree"
        python SimpleTree.py
        ;;
        t ) echo "===> split tree"
        python SplitTree.py
        ;;
        u ) echo "===> make unet data"
        root -b -q MakeUNetData.cc
        ;;
        m ) echo "===> train unet"
        python DenoisingUNet.py
        ;;
        p ) echo "===> make plots"
        root -b -q PlotResults.cc
        ;;
        \? ) echo "use cmds
        -s make simple tree
        -t spilt tree randomly
        -u make unet data
        -m train the model
        -p plot the results
        "
        ;;
    esac
done
