/*
 * dinupa3@gmail.com
 */

#include <TFile.h>
#include <TTree.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <iostream>

#include "MakeHist.h"
#include "MakeTree.h"

int MakeUNetData()
{

    auto event = new TRandom3();

    /*
     * apply simple cuts
     */
    // gSystem->Exec("python SimpleTree.py");


    /*
     * split events randomly
     */
    // gSystem->Exec("python SplitTree.py");

    /*
     * make tree with histograms
     */
    auto train_mh = new MakeHist();
    train_mh->Init("train_data");

    auto val_mh = new MakeHist();
    val_mh->Init("val_data");

    auto test_mh = new MakeHist();
    test_mh->Init("test_data");

    auto outfite = new TFile("unet.root", "RECREATE");

    auto train_tree = new MakeTree();
    train_tree->Init("train_tree");

    cout << "===> create train tree <===" << endl;
    train_tree->FillTree(train_mh, 70000, event);

    auto val_tree = new MakeTree();
    val_tree->Init("val_tree");

    cout << "===> create val tree <===" << endl;
    val_tree->FillTree(val_mh, 30000, event);

    auto test_tree = new MakeTree();
    test_tree->Init("test_tree");

    cout << "===> create test tree <===" << endl;
    test_tree->FillTree(test_mh, 40000, event);

    train_tree->tree->Write();
    val_tree->tree->Write();
    test_tree->tree->Write();
    outfite->Close();

    /*
     * save to torch tensor
     */
    // gSystem->Exec("python SaveTensor.py");

    return 0;

}
