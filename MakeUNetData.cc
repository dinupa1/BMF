//
// dinupa3@gmail.com
//

#include <TFile.h>
#include <TTree.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TCanvas.h>
#include <iostream>

#include "MakeTree.h"

int MakeUNetData()
{
    //gStyle->SetOptStat(0);

    auto train_mh = new MakeHist();
    train_mh->Init("train_data");

    auto val_mh = new MakeHist();
    val_mh->Init("val_data");

    auto test_mh = new MakeHist();
    test_mh->Init("test_data");

    auto outfite = new TFile("unet.root", "RECREATE");

    auto train_tree = new MakeTree();
    train_tree->Init("train_tree");

    cout << "*** create train tree ***" << endl;
    train_tree->FillTree(train_mh, 1000);

    auto val_tree = new MakeTree();
    val_tree->Init("val_tree");

    cout << "*** create val tree ***" << endl;
    val_tree->FillTree(val_mh, 1000);

    auto test_tree = new MakeTree();
    test_tree->Init("test_tree");

    cout << "*** create test tree ***" << endl;
    test_tree->FillTree(test_mh, 1000);

    train_tree->tree->Write();
    val_tree->tree->Write();
    test_tree->tree->Write();

    outfite->Close();

    return 0;

}
