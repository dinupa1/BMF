#include <TFile.h>
#include <TTree.h>
#include <TString.h>
#include <TH1D.h>
#include <TSystem.h>
#include <iostream>

#include "CleanTree.hh"


#define DEBUG

void CleanTree()
{

    TFile* merged_RS67_LH2 = TFile::Open("merged_RS67_3089LH2.root", "READ");
    TTree* tree = (TTree*)merged_RS67_LH2->Get("result");
    TTree* tree_mix = (TTree*)merged_RS67_LH2->Get("result_mix");

    TFile* merged_RS67_flask = TFile::Open("merged_RS67_3089flask.root", "READ");
    TTree* tree_flask = (TTree*)merged_RS67_flask->Get("result");


    TFile* outfile = new TFile("simple_tree.root", "RECREATE");

    TTree* tree_cuts = (TTree*)tree->CopyTree(all_cuts.Data());
    tree_cuts->SetName("tree");
    TTree* tree_mix_cuts = (TTree*)tree_mix->CopyTree(all_cuts.Data());
    tree_mix_cuts->SetName("tree_mix");
    TTree* tree_flask_cuts = (TTree*)tree_flask->CopyTree(all_cuts.Data());
    tree_flask_cuts->SetName("tree_flask");


#ifdef DEBUG
    std::cout << "---> tree entries after all the cuts = " << tree->GetEntries(all_cuts.Data()) << std::endl;

    std::cout << "---> mix tree entries after all the cuts = " << tree_mix->GetEntries(all_cuts.Data()) << std::endl;

    std::cout << "---> flask tree entries after all the cuts = " << tree_flask->GetEntries(all_cuts.Data()) << std::endl;
#endif

    outfile->cd();

    tree_cuts->Write();
    tree_mix_cuts->Write();
    tree_flask_cuts->Write();

    outfile->Close();

    gSystem->Exec("python ./src/data_for_reweighting.py");

}


