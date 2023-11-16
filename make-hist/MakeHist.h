/*
 * dinupa3@gmail.com
 */

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

    static const int num_bins = mass_bins* pT_bins* xF_bins* phi_bins* costh_bins;

    double mass_edges[mass_bins+1] = {4.0, 5.5, 6.5, 10.0};
    double pT_edges[pT_bins+1] = {0.0, 0.5, 1.5, 3.5};
    double xF_edges[xF_bins+1] = {-0.2, 0.1, 0.4, 0.7, 1.0};
    double phi_edges[phi_bins+1];
    double costh_edges[costh_bins+1];

    void fill_true_phi_costh(int i, int j, int k, float event_weight);
    void fill_reco_phi_costh(int i, int j, int k, float event_weight);

public:
    int num_events;
    float true_count[num_bins];
    float true_error2[num_bins];
    float reco_count[num_bins];
    float reco_error2[num_bins];

    MakeHist();
    virtual ~MakeHist(){};
    void Init(TString tree_name);
    void FillHist(float lambda, float mu, float nu, TRandom3* event);
};


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
    for(int ii = 0; ii < phi_bins; ii+=3)
    {
        for(int jj = 0; jj < costh_bins; jj+=3)
        {
            // ii = 0
            if(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1] &&
                pT_edges[j] < true_pT && true_pT <= pT_edges[j+1] &&
                xF_edges[k] < true_xF && true_xF <= xF_edges[k+1] &&
                phi_edges[ii+0] < true_phi && true_phi <= phi_edges[ii+0+1] &&
                costh_edges[jj+0] < true_costh && true_costh <= costh_edges[ii+0+1])
                {
                    true_count[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+0)] += event_weight;
                    true_error2[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+0)] += event_weight* event_weight;
                }

            if(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1] &&
                pT_edges[j] < true_pT && true_pT <= pT_edges[j+1] &&
                xF_edges[k] < true_xF && true_xF <= xF_edges[k+1] &&
                phi_edges[ii+0] < true_phi && true_phi <= phi_edges[ii+0+1] &&
                costh_edges[jj+1] < true_costh && true_costh <= costh_edges[ii+1+1])
                {
                    true_count[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+1)] += event_weight;
                    true_error2[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+1)] += event_weight* event_weight;
                }

            if(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1] &&
                pT_edges[j] < true_pT && true_pT <= pT_edges[j+1] &&
                xF_edges[k] < true_xF && true_xF <= xF_edges[k+1] &&
                phi_edges[ii+0] < true_phi && true_phi <= phi_edges[ii+0+1] &&
                costh_edges[jj+2] < true_costh && true_costh <= costh_edges[ii+2+1])
                {
                    true_count[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+2)] += event_weight;
                    true_error2[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+2)] += event_weight* event_weight;
                }

            // ii = 1
            if(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1] &&
                pT_edges[j] < true_pT && true_pT <= pT_edges[j+1] &&
                xF_edges[k] < true_xF && true_xF <= xF_edges[k+1] &&
                phi_edges[ii+1] < true_phi && true_phi <= phi_edges[ii+1+1] &&
                costh_edges[jj+0] < true_costh && true_costh <= costh_edges[ii+0+1])
                {
                    true_count[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+0)] += event_weight;
                    true_error2[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+0)] += event_weight* event_weight;
                }

            if(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1] &&
                pT_edges[j] < true_pT && true_pT <= pT_edges[j+1] &&
                xF_edges[k] < true_xF && true_xF <= xF_edges[k+1] &&
                phi_edges[ii+1] < true_phi && true_phi <= phi_edges[ii+1+1] &&
                costh_edges[jj+1] < true_costh && true_costh <= costh_edges[ii+1+1])
                {
                    true_count[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+1)] += event_weight;
                    true_error2[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+1)] += event_weight* event_weight;
                }

            if(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1] &&
                pT_edges[j] < true_pT && true_pT <= pT_edges[j+1] &&
                xF_edges[k] < true_xF && true_xF <= xF_edges[k+1] &&
                phi_edges[ii+1] < true_phi && true_phi <= phi_edges[ii+1+1] &&
                costh_edges[jj+2] < true_costh && true_costh <= costh_edges[ii+2+1])
                {
                    true_count[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+2)] += event_weight;
                    true_error2[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+2)] += event_weight* event_weight;
                }

            // ii = 2
            if(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1] &&
                pT_edges[j] < true_pT && true_pT <= pT_edges[j+1] &&
                xF_edges[k] < true_xF && true_xF <= xF_edges[k+1] &&
                phi_edges[ii+2] < true_phi && true_phi <= phi_edges[ii+2+1] &&
                costh_edges[jj+0] < true_costh && true_costh <= costh_edges[ii+0+1])
                {
                    true_count[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+0)] += event_weight;
                    true_error2[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+0)] += event_weight* event_weight;
                }

            if(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1] &&
                pT_edges[j] < true_pT && true_pT <= pT_edges[j+1] &&
                xF_edges[k] < true_xF && true_xF <= xF_edges[k+1] &&
                phi_edges[ii+2] < true_phi && true_phi <= phi_edges[ii+2+1] &&
                costh_edges[jj+1] < true_costh && true_costh <= costh_edges[ii+1+1])
                {
                    true_count[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+1)] += event_weight;
                    true_error2[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+1)] += event_weight* event_weight;
                }

            if(mass_edges[i] < true_mass && true_mass <= mass_edges[i+1] &&
                pT_edges[j] < true_pT && true_pT <= pT_edges[j+1] &&
                xF_edges[k] < true_xF && true_xF <= xF_edges[k+1] &&
                phi_edges[ii+2] < true_phi && true_phi <= phi_edges[ii+2+1] &&
                costh_edges[jj+2] < true_costh && true_costh <= costh_edges[ii+2+1])
                {
                    true_count[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+2)] += event_weight;
                    true_error2[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+2)] += event_weight* event_weight;
                }
        }
    }
}


void MakeHist::fill_reco_phi_costh(int i, int j, int k, float event_weight)
{
    for(int ii = 0; ii < phi_bins; ii+=3)
    {
        for(int jj = 0; jj < costh_bins; jj+=3)
        {
            // ii = 0
            if(mass_edges[i] < mass && mass <= mass_edges[i+1] &&
                pT_edges[j] < pT && pT <= pT_edges[j+1] &&
                xF_edges[k] < xF && xF <= xF_edges[k+1] &&
                phi_edges[ii+0] < phi && phi <= phi_edges[ii+0+1] &&
                costh_edges[jj+0] < costh && costh <= costh_edges[ii+0+1])
                {
                    reco_count[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+0)] += event_weight;
                    reco_error2[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+0)] += event_weight* event_weight;
                }

            if(mass_edges[i] < mass && mass <= mass_edges[i+1] &&
                pT_edges[j] < pT && pT <= pT_edges[j+1] &&
                xF_edges[k] < xF && xF <= xF_edges[k+1] &&
                phi_edges[ii+0] < phi && phi <= phi_edges[ii+0+1] &&
                costh_edges[jj+1] < costh && costh <= costh_edges[ii+1+1])
                {
                    reco_count[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+1)] += event_weight;
                    reco_error2[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+1)] += event_weight* event_weight;
                }

            if(mass_edges[i] < mass && mass <= mass_edges[i+1] &&
                pT_edges[j] < pT && pT <= pT_edges[j+1] &&
                xF_edges[k] < xF && xF <= xF_edges[k+1] &&
                phi_edges[ii+0] < phi && phi <= phi_edges[ii+0+1] &&
                costh_edges[jj+2] < costh && costh <= costh_edges[ii+2+1])
                {
                    reco_count[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+2)] += event_weight;
                    reco_error2[1728* i+ 576* j+ 144* k+ 12* (ii+0) + (jj+2)] += event_weight* event_weight;
                }

            // ii = 1
            if(mass_edges[i] < mass && mass <= mass_edges[i+1] &&
                pT_edges[j] < pT && pT <= pT_edges[j+1] &&
                xF_edges[k] < xF && xF <= xF_edges[k+1] &&
                phi_edges[ii+1] < phi && phi <= phi_edges[ii+1+1] &&
                costh_edges[jj+0] < costh && costh <= costh_edges[ii+0+1])
                {
                    reco_count[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+0)] += event_weight;
                    reco_error2[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+0)] += event_weight* event_weight;
                }

            if(mass_edges[i] < mass && mass <= mass_edges[i+1] &&
                pT_edges[j] < pT && pT <= pT_edges[j+1] &&
                xF_edges[k] < xF && xF <= xF_edges[k+1] &&
                phi_edges[ii+1] < phi && phi <= phi_edges[ii+1+1] &&
                costh_edges[jj+1] < costh && costh <= costh_edges[ii+1+1])
                {
                    reco_count[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+1)] += event_weight;
                    reco_error2[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+1)] += event_weight* event_weight;
                }

            if(mass_edges[i] < mass && mass <= mass_edges[i+1] &&
                pT_edges[j] < pT && pT <= pT_edges[j+1] &&
                xF_edges[k] < xF && xF <= xF_edges[k+1] &&
                phi_edges[ii+1] < phi && phi <= phi_edges[ii+1+1] &&
                costh_edges[jj+2] < costh && costh <= costh_edges[ii+2+1])
                {
                    reco_count[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+2)] += event_weight;
                    reco_error2[1728* i+ 576* j+ 144* k+ 12* (ii+1) + (jj+2)] += event_weight* event_weight;
                }

            // ii = 2
            if(mass_edges[i] < mass && mass <= mass_edges[i+1] &&
                pT_edges[j] < pT && pT <= pT_edges[j+1] &&
                xF_edges[k] < xF && xF <= xF_edges[k+1] &&
                phi_edges[ii+2] < phi && phi <= phi_edges[ii+2+1] &&
                costh_edges[jj+0] < costh && costh <= costh_edges[ii+0+1])
                {
                    reco_count[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+0)] += event_weight;
                    reco_error2[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+0)] += event_weight* event_weight;
                }

            if(mass_edges[i] < mass && mass <= mass_edges[i+1] &&
                pT_edges[j] < pT && pT <= pT_edges[j+1] &&
                xF_edges[k] < xF && xF <= xF_edges[k+1] &&
                phi_edges[ii+2] < phi && phi <= phi_edges[ii+2+1] &&
                costh_edges[jj+1] < costh && costh <= costh_edges[ii+1+1])
                {
                    reco_count[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+1)] += event_weight;
                    reco_error2[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+1)] += event_weight* event_weight;
                }

            if(mass_edges[i] < mass && mass <= mass_edges[i+1] &&
                pT_edges[j] < pT && pT <= pT_edges[j+1] &&
                xF_edges[k] < xF && xF <= xF_edges[k+1] &&
                phi_edges[ii+2] < phi && phi <= phi_edges[ii+2+1] &&
                costh_edges[jj+2] < costh && costh <= costh_edges[ii+2+1])
                {
                    reco_count[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+2)] += event_weight;
                    reco_error2[1728* i+ 576* j+ 144* k+ 12* (ii+2) + (jj+2)] += event_weight* event_weight;
                }
        }
    }
}


void MakeHist::FillHist(float lambda, float mu, float nu, TRandom3* event)
{
    int reco_events = 10000;
    int n_reco = 0;

    for(int i = 0; i < num_bins; i++)
    {
        true_count[i] = 0.0;
        true_error2[i] = 0.0;
        reco_count[i] = 0.0;
        reco_error2[i] = 0.0;
    }

    for(int i = 0; i < num_events; i++)
    {
        data->GetEntry(i);

        float mass_rand = event->Uniform(4., 10.);
        float pT_rand = event->Uniform(0., 3.5);
        float xF_rand =event->Uniform(-0.2, 1.0);

        if(mass_rand <= true_mass && pT_rand <= true_pT && xF_rand <= true_xF)
        {
            double event_weight = weight_fn(lambda, mu, nu, true_phi, true_costh);

            // particle level data
            for(int mm = 0; mm < mass_bins; mm++)
            {
                for(int nn = 0; nn < pT_bins; nn++)
                {
                    for(int pp = 0; pp < xF_bins; pp++)
                    {
                        fill_true_phi_costh(mm, nn, pp, event_weight);
                    }
                }
            }

            // detector level data
            if(fpga1==1 && mass > 0.)
            {
                for(int mm = 0; mm < mass_bins; mm++)
                {
                    for(int nn = 0; nn < pT_bins; nn++)
                    {
                        for(int pp = 0; pp < xF_bins; pp++)
                        {
                            fill_reco_phi_costh(mm, nn, pp, event_weight);
                        }
                    }
                }
                n_reco += 1;
            }
        }

        if(n_reco == reco_events)
        {
            cout << "===> Filled hitogram with " << n_reco << " events <===" << endl;
            break;
        }
    }
}
#endif /* _H_MakeHist_H_ */
