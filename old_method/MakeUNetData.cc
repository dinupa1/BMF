//
// dinupa3@gmail.com
//

#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TMath.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TRandom.h>
#include <TCanvas.h>
#include <iostream>

#include "MakeTree.h"

using namespace std;

double weight_fn(double lambda, double mu, double nu, double phi, double costh)
{
    double weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
    return weight/(1. + costh* costh);
}


MakeHist::MakeHist()
{;}


void MakeHist::Init(TString tree_name)
{
    TFile* file = TFile::Open("split.root", "READ");
    data = (TTree*)file->Get(tree_name.Data());
    events = data->GetEntries();

    data->SetBranchAddress("fpga1",         &fpga1);
    data->SetBranchAddress("mass",          &mass);
    data->SetBranchAddress("pT",            &pT);
    data->SetBranchAddress("xF",            &xF);
    data->SetBranchAddress("phi",           &phi);
    data->SetBranchAddress("costh",         &costh);
    data->SetBranchAddress("true_mass",     &true_mass);
    data->SetBranchAddress("true_pT",       &true_pT);
    data->SetBranchAddress("true_xF",       &true_xF);
    data->SetBranchAddress("true_phi",      &true_phi);
    data->SetBranchAddress("true_costh",    &true_costh);
}


void MakeHist::FillHist(int ev1, int ev2, double lambda, double mu, double nu)
{
    true_hist = new TH2D("true_hist", "true_hist", phi_bins, -pi, pi, costh_bins, -0.6, 0.6);
    reco_hist = new TH2D("reco_hist", "reco_hist", phi_bins, -pi, pi, costh_bins, -0.6, 0.6);

    true_hist->Sumw2();
    reco_hist->Sumw2();

    for(int i = ev1; i < ev2; i++)
    {
        data->GetEntry(i);
        true_hist->Fill(true_phi, true_costh, weight_fn(lambda, mu, nu, true_phi, true_costh));

        if(fpga1 == 1 && mass > 0.)
        {
            reco_hist->Fill(phi, costh, weight_fn(lambda, mu, nu, true_phi, true_costh));
        }
    }

    // double integral_value = reco_hist->Integral();
    //
    // reco_hist->Scale(1./integral_value, "WIDTH");
    // true_hist->Scale(1./integral_value, "WIDTH");
    //
    // cout << "Integral value = " << integral_value << endl;

    // TCanvas* can = new TCanvas();
    //
    // true_hist->Draw("COLZ");
    // can->SaveAs("imgs/true_hist.png");
    //
    // reco_hist->Draw("COLZ");
    // can->SaveAs("imgs/reco_hist.png");
}


MakeTree::MakeTree()
{;}

void MakeTree::Init(TString tree_name, int n)
{
    rn = new TRandom(n);

    tree = new TTree(tree_name.Data(), tree_name.Data());
    tree->Branch("true_hist",       true_hist,      "true_hist[2][12][12]/D");
    tree->Branch("reco_hist",       reco_hist,      "reco_hist[2][12][12]/D");
    tree->Branch("lambda",          &lambda,        "lambda/D");
    tree->Branch("mu",              &mu,            "mu/D");
    tree->Branch("nu",              &nu,            "nu/D");
}


void MakeTree::FillTree(MakeHist* mh, int n_events)
{
    for(int i = 0; i < n_events; i++)
    {
        int event = TMath::Nint(rn->Uniform(0, (mh->events)/2));

        lambda = rn->Uniform(-1., 1.);
        mu = rn->Uniform(-0.5, 0.5);
        nu = rn->Uniform(-0.5, 0.5);

        mh->FillHist(event, event+h_events, lambda, mu, nu);

        for(int i = 0; i < mh->phi_bins; i++)
        {
            for(int j = 0; j < mh->costh_bins; j++)
            {
                true_hist[0][i][j] = mh->true_hist->GetBinContent(i+1, j+1) - mh->true_hist->GetBinError(i+1, j+1);
                true_hist[1][i][j] = mh->true_hist->GetBinContent(i+1, j+1) + mh->true_hist->GetBinError(i+1, j+1);

                reco_hist[0][i][j] = mh->reco_hist->GetBinContent(i+1, j+1) - mh->reco_hist->GetBinError(i+1, j+1);
                reco_hist[1][i][j] = mh->reco_hist->GetBinContent(i+1, j+1) + mh->reco_hist->GetBinError(i+1, j+1);
            }
        }

        if (i%10000==0) {cout << "===> event : " << event << " lambda : " << lambda << " mu : " << mu << " nu : " << nu << endl;}

        tree->Fill();

        delete mh->true_hist;
        delete mh->reco_hist;
    }
}



void MakeUNetData()
{
    //gStyle->SetOptStat(0);

    auto train_mh = new MakeHist();
    train_mh->Init("train_data");

    auto val_mh = new MakeHist();
    val_mh->Init("test_data");

    auto outfite = new TFile("unet.root", "RECREATE");

    auto train_tree = new MakeTree();
    train_tree->Init("train_tree", 2);

    cout << "*** create train tree ***" << endl;
    train_tree->FillTree(train_mh, 100000);

    auto val_tree = new MakeTree();
    val_tree->Init("val_tree", 3);

    cout << "*** create val tree ***" << endl;
    val_tree->FillTree(val_mh, 40000);

    train_tree->tree->Write();
    val_tree->tree->Write();

    outfite->Close();

}
