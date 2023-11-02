//
// dinupa3@gmail.com
//

#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TMath.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TRandom3.h>
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


void MakeHist::FillHist(double lambda, double mu, double nu, int seed)
{
    TRandom3* acceptance = new TRandom3(seed);

    double mass_edges[4] = {4.0, 5.5, 6.5, 10.0};
    double pT_edges[4] = {0.0, 0.5, 1.5, 3.5};
    double xF_edges[5] = {-0.2, 0.1, 0.4, 0.7, 1.0};
    double phi_edges[13];
    double costh_edges[13];

    for(int i = 0; i < 13; i++){phi_edges[i] = -pi + i* (pi - (-pi))/12;}
    for(int i = 0; i < 13; i++){costh_edges[i] = -0.6 + i* (0.6 - (-0.6))/12;}

    int reco_events = 10000;
    int n_reco = 0;

    for(int ii = 0; ii < events; ii++)
    {
        data->GetEntry(ii);

        double mass_rand = acceptance->Uniform(4., 10.);
        double pT_rand = acceptance->Uniform(0.0, 3.5);
        double xF_rand = acceptance->Uniform(-0.2, 1.0);

        if(sqrt(mass_rand* mass_rand + pT_rand* pT_rand + xF_rand* xF_rand) <= sqrt(true_mass* true_mass + true_pT* true_pT + true_xF* true_xF))
        {
            double event_weight = weight_fn(lambda, mu, nu, true_phi, true_costh);

            double bin_error1 = 0;
            double bin_error2 = 0;

            // particle level information
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    for(int k = 0; k < 4; k++)
                    {
                        for(int l = 0; l < 12; l++)
                        {
                            for(int m = 0; m < 12; m++)
                            {
                                if(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1] && pT_edges[j] < true_pT && true_pT <= pT_edges[j+1] && xF_edges[k] < true_xF && true_xF <= xF_edges[k+1] && phi_edges[l] < true_phi && true_phi <= phi_edges[l+1] && costh_edges[m] < true_costh && true_costh <= costh_edges[m+1])
                                {
                                    true_count[i][j][k][l][m] += event_weight;
                                    bin_error1 += event_weight* event_weight;
                                    true_error[i][j][k][l][m] = sqrt(bin_error1);
                                }
                            }
                        }
                    }
                }
            }

            if(fpga1==1 && mass > 0.)
            {
                // detector level information
                for(int i = 0; i < 3; i++)
                {
                    for(int j = 0; j < 3; j++)
                    {
                        for(int k = 0; k < 4; k++)
                        {
                            for(int l = 0; l < 12; l++)
                            {
                                for(int m = 0; m < 12; m++)
                                {
                                    if(mass_edges[i] < mass && mass <= mass_edges[i+1] && pT_edges[j] < pT && pT <= pT_edges[j+1] && xF_edges[k] < xF && xF <= xF_edges[k+1] && phi_edges[l] < phi && phi <= phi_edges[l+1] && costh_edges[m] < costh && costh <= costh_edges[m+1])
                                    {
                                        reco_count[i][j][k][l][m] += event_weight;
                                        bin_error2 += event_weight* event_weight;
                                        reco_error[i][j][k][l][m] = sqrt(bin_error2);
                                    }
                                }
                            }
                        }
                    }
                }
                n_reco += 1;
            }
            if(n_reco == reco_events)
            {
                //cout << "filled with " << n_reco << " reco. events " << endl;
                // cout << "seed " << seed << endl;
                break;
            }
        }
    }
}


MakeTree::MakeTree()
{;}

void MakeTree::Init(TString tree_name)
{
    tree = new TTree(tree_name.Data(), tree_name.Data());
    tree->Branch("true_hist",       true_hist,      "true_hist[2][72][72]/D");
    tree->Branch("reco_hist",       reco_hist,      "reco_hist[2][72][72]/D");
    tree->Branch("lambda",          &lambda,        "lambda/D");
    tree->Branch("mu",              &mu,            "mu/D");
    tree->Branch("nu",              &nu,            "nu/D");
}


void MakeTree::FillTree(MakeHist* mh, int n_events, int seed)
{
    TRandom3* theta = new TRandom3(seed);

    for(int i = 0; i < n_events; i++)
    {
        lambda = theta->Uniform(-1., 1.);
        mu = theta->Uniform(-0.5, 0.5);
        nu = theta->Uniform(-0.5, 0.5);

        mh->FillHist(lambda, mu, nu, TMath::Nint(theta->Uniform(0., 100.)));

        double true_count[5184];
        double true_error[5184];
        double reco_count[5184];
        double reco_error[5184];

        // make data structure
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                for(int k = 0; k < 4; k++)
                {
                    for(int l = 0; l < 12; l++)
                    {
                        for(int m = 0; m < 12; m++)
                        {
                            true_count[1728* i+ 576* j+ 144* k+ 12* l + m] = mh->true_count[i][j][k][l][m];
                            true_error[1728* i+ 576* j+ 144* k+ 12* l + m] = mh->true_error[i][j][k][l][m];

                            reco_count[1728* i+ 576* j+ 144* k+ 12* l + m] = mh->reco_count[i][j][k][l][m];
                            reco_error[1728* i+ 576* j+ 144* k+ 12* l + m] = mh->reco_error[i][j][k][l][m];
                        }
                    }
                }
            }
        }

        for(int i = 0; i < 72; i++)
        {
            for(int j = 0; j < 72; j++)
            {
                true_hist[0][i][j] = true_count[72*i + j] - true_error[72*i + j];
                true_hist[1][i][j] = true_count[72*i + j] + true_error[72*i + j];

                reco_hist[0][i][j] = reco_count[72*i + j] - reco_error[72*i + j];
                reco_hist[1][i][j] = reco_count[72*i + j] + reco_error[72*i + j];
            }
        }

        if (i+1%10000==0) {cout << "===> event : " << i+1 << " lambda : " << lambda << " mu : " << mu << " nu : " << nu << endl;}

        tree->Fill();
    }
}



int main()
{
    //gStyle->SetOptStat(0);

    auto train_mh = new MakeHist();
    train_mh->Init("train_data");

    auto val_mh = new MakeHist();
    val_mh->Init("val_data");

    auto test_mh = new MakeHist();
    test_mh->Init("test_data");

    TRandom3* events = new TRandom3();

    auto outfite = new TFile("unet.root", "RECREATE");

    auto train_tree = new MakeTree();
    train_tree->Init("train_tree");

    cout << "*** create train tree ***" << endl;
    train_tree->FillTree(train_mh, 60000, TMath::Nint(events->Uniform(0., 100000.)));

    auto val_tree = new MakeTree();
    val_tree->Init("val_tree");

    cout << "*** create val tree ***" << endl;
    val_tree->FillTree(val_mh, 40000, TMath::Nint(events->Uniform(100000., 200000.)));

    auto test_tree = new MakeTree();
    test_tree->Init("test_tree");

    cout << "*** create test tree ***" << endl;
    test_tree->FillTree(test_mh, 30000, TMath::Nint(events->Uniform(200000., 300000.)));

    train_tree->tree->Write();
    val_tree->tree->Write();
    test_tree->tree->Write();

    outfite->Close();

    return 0;

}
