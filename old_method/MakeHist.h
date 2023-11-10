//
// dinupa3@gmail.com
//

#include <TFile.h>
#include <TTree.h>
#include <TMath.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TRandom3.h>
#include <iostream>

using namespace std;

float weight_fn(float lambda, float mu, float nu, float phi, float costh)
{
    float weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
    return weight/(1. + costh* costh);
}

auto event = new TRandom3();

#ifndef _H_MakeHist_H_
#define _H_MakeHist_H_

class MakeHist
{
    TTree* data;
    int fpga1;
    float mass, pT, xF, phi, costh, true_mass, true_pT, true_xF, true_phi, true_costh;
    double pi = TMath::Pi();

    static const int mass_bins = 3;
    static const int pT_bins = 3;
    static const int xF_bins = 4;
    static const int phi_bins = 12;
    static const int costh_bins = 12;

    double mass_edges[mass_bins+1] = {4.0, 5.5, 6.5, 10.0};
    double pT_edges[pT_bins+1] = {0.0, 0.5, 1.5, 3.5};
    double xF_edges[xF_bins+1] = {-0.2, 0.1, 0.4, 0.7, 1.0};
    double phi_edges[phi_bins+1];
    double costh_edges[costh_bins+1];

    void fill_true_phi_costh(int i, int j, int k, float event_weight);
    void fill_reco_phi_costh(int i, int j, int k, float event_weight);
    void fill_true_mass_pT_xF(float event_weight);
    void fill_reco_mass_pT_xF(float event_weight);

public:
    int num_events;
    float true_count[mass_bins][pT_bins][xF_bins][phi_bins][costh_bins];
    float true_error2[mass_bins][pT_bins][xF_bins][phi_bins][costh_bins];
    float reco_count[mass_bins][pT_bins][xF_bins][phi_bins][costh_bins];
    float reco_error2[mass_bins][pT_bins][xF_bins][phi_bins][costh_bins];

    MakeHist();
    virtual ~MakeHist(){};
    void Init(TString tree_name);
    void FillHist(float lambda, float mu, float nu);
};
#endif /* _H_MakeHist_H_ */


MakeHist::MakeHist()
{;}


void MakeHist::Init(TString tree_name)
{
    TFile* file = TFile::Open("split.root", "READ");
    data = (TTree*)file->Get(tree_name.Data());
    num_events = data->GetEntries();

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

    for(int i = 0; i < phi_bins+1; i++){phi_edges[i] = -pi + i* (pi - (-pi))/phi_bins;}
    for(int i = 0; i < costh_bins+1; i++){costh_edges[i] = -0.6 + i* (0.6 - (-0.6))/costh_bins;}
}


void MakeHist::fill_true_phi_costh(int i, int j, int k, float event_weight)
{
    for(int ii = 0; ii < phi_bins; ii++)
    {
        if(!(phi_edges[ii] < true_phi && true_phi <= phi_edges[ii+1])){continue;}
        for(int jj = 0; jj < costh_bins; jj++)
        {
            if(!(costh_edges[jj] < true_costh && true_costh <= costh_edges[jj+1])){continue;}
            true_count[i][j][k][ii][jj] += event_weight;
            true_error2[i][j][k][ii][jj] += event_weight* event_weight;
        }
    }
}


void MakeHist::fill_reco_phi_costh(int i, int j, int k, float event_weight)
{
    for(int ii = 0; ii < phi_bins; ii++)
    {
        if(!(phi_edges[ii] < phi && phi <= phi_edges[ii+1])){continue;}
        for(int jj = 0; jj < costh_bins; jj++)
        {
            if(!(costh_edges[jj] < costh && costh <= costh_edges[jj+1])){continue;}
            reco_count[i][j][k][ii][jj] += event_weight;
            reco_error2[i][j][k][ii][jj] += event_weight* event_weight;
        }
    }
}


void MakeHist::fill_true_mass_pT_xF(float event_weight)
{
    for(int i = 0; i < mass_bins; i++)
    {
        if(!(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1])){continue;}
        for(int j = 0; j < pT_bins; j++)
        {
            if(!(pT_edges[j] < true_pT && true_pT <= pT_edges[j+1])){continue;}
            for(int k = 0; k < xF_bins ; k++)
            {
                if(!(xF_edges[k] < true_xF && true_xF <= xF_edges[k+1])){continue;}
                fill_true_phi_costh(i, j, k, event_weight);
            }
        }
    }
}



void MakeHist::fill_reco_mass_pT_xF(float event_weight)
{
    for(int i = 0; i < mass_bins; i++)
    {
        if(!(mass_edges[i] < mass && mass <= mass_edges[i+1])){continue;}
        for(int j = 0; j < pT_bins; j++)
        {
            if(!(pT_edges[j] < pT && pT <= pT_edges[j+1])){continue;}
            for(int k = 0; k < xF_bins ; k++)
            {
                if(!(xF_edges[k] < xF && xF <= xF_edges[k+1])){continue;}
                fill_reco_phi_costh(i, j, k, event_weight);
            }
        }
    }
}


void MakeHist::FillHist(float lambda, float mu, float nu)
{
    int reco_events = 10000;
    int n_reco = 0;

    for(int i = 0; i < num_events; i++)
    {
        data->GetEntry(i);

        float mass_rand = event->Uniform(4., 10.);
        float pT_rand = event->Uniform(0., 3.5);
        float xF_rand =event->Uniform(-0.2, 1.0);

        float radius_rand = sqrt(mass_rand* mass_rand + pT_rand* pT_rand + xF_rand* xF_rand);
        float radius_true = sqrt(true_mass* true_mass + true_pT* true_pT + true_xF* true_xF);

        if(!(mass_rand <= true_mass)){continue;} // rejection sampling method
        float event_weight = weight_fn(lambda, mu, nu, true_phi, true_costh);

        // particle level info.
        fill_true_mass_pT_xF(event_weight);

        // detector level info.
        if(!(fpga1 == 1 && mass > 0.)){continue;}
        fill_reco_mass_pT_xF(event_weight);
        n_reco += 1;

        if(n_reco == reco_events){break;}
    }
}
