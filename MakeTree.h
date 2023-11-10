//
// dinupa3@gmail.com
//

#include <TFile.h>
#include <TTree.h>
#include <TMath.h>
#include <TCanvas.h>
#include <iostream>

#include "MakeHist.h"

using namespace std;

auto theta = new TRandom3();

#ifndef _H_MakeTree_H_
#define _H_MakeTree_H_

class MakeTree
{
    float true_hist[2][72][72];
    float reco_hist[2][72][72];
    float lambda, mu, nu;

    float true_count[5184];
    float true_error[5184];
    float reco_count[5184];
    float reco_error[5184];

    void read_phi_costh( MakeHist* mh, int i, int j, int k);
    void read_mass_pT_xF(MakeHist* mh);
    void fill_hist();
public:
    TTree* tree;
    MakeTree();
    virtual ~MakeTree(){};
    void Init(TString tree_name);
    void FillTree(MakeHist* mh, int n_events);
};
#endif /* _H_MakeTree_H_ */

MakeTree::MakeTree()
{;}

void MakeTree::Init(TString tree_name)
{
    tree = new TTree(tree_name.Data(), tree_name.Data());
    tree->Branch("true_hist",       true_hist,      "true_hist[2][72][72]/F");
    tree->Branch("reco_hist",       reco_hist,      "reco_hist[2][72][72]/F");
    tree->Branch("lambda",          &lambda,        "lambda/F");
    tree->Branch("mu",              &mu,            "mu/F");
    tree->Branch("nu",              &nu,            "nu/F");
}


void MakeTree::read_phi_costh( MakeHist* mh, int i, int j, int k)
{
    for(int l = 0; l < 12; l++)
    {
        for(int m = 0; m < 12; m++)
        {
            true_count[1728* i+ 576* j+ 144* k+ 12* l + m] = mh->true_count[i][j][k][l][m];
            true_error[1728* i+ 576* j+ 144* k+ 12* l + m] = sqrt(mh->true_error2[i][j][k][l][m]);
            reco_count[1728* i+ 576* j+ 144* k+ 12* l + m] = mh->reco_count[i][j][k][l][m];
            reco_error[1728* i+ 576* j+ 144* k+ 12* l + m] = sqrt(mh->reco_error2[i][j][k][l][m]);
        }
    }
}


void MakeTree::read_mass_pT_xF(MakeHist* mh)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            for(int k = 0; k < 4; k++)
            {
                read_phi_costh(mh, i, j, k);
            }
        }
    }
}


void MakeTree::fill_hist()
{
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
}


void MakeTree::FillTree(MakeHist* mh, int n_events)
{
    for(int i = 0; i < n_events; i++)
    {
        lambda = theta->Uniform(-1., 1.);
        mu = theta->Uniform(-0.5, 0.5);
        nu = theta->Uniform(-0.5, 0.5);

        mh->FillHist(lambda, mu, nu);

        read_mass_pT_xF(mh);

        fill_hist();

        if (i+1%100==0) {cout << "===> event : " << i+1 << " lambda : " << lambda << " mu : " << mu << " nu : " << nu << endl;}
        // cout << "===> event : " << i << " lambda : " << lambda << " mu : " << mu << " nu : " << nu << endl;

        tree->Fill();
    }
}
