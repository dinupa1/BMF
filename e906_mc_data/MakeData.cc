/*
 * dinupa3@gmail.com
 *
 */
#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TString.h>
#include <TH2D.h>
#include <iostream>

#include "TreeData.hh"

using namespace std;

TString fname = "split.root";


double lambda_secret = 0.8;
double mu_secret = 0.1;
double nu_secret = 0.2;

double weight_fn(double lambda, double mu, double nu, double phi, double costh)
{
    double weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
    return weight/(1. + costh* costh);
}


void make_tree(TString tname)
{
    save = new TTree(tname.Data(), tname.Data());

    save->Branch("thetas",      &thetas,        "thetas[3]/D");
    save->Branch("X_par",       &X_par,         "X_par[2]/D");
    save->Branch("X_det",       &X_det,         "X_det[2]/D");
    save->Branch("W_par",       &W_par,         "W_par[2]/D");
    save->Branch("W_det",       &W_det,         "W_det[2]/D");
    save->Branch("label",       &label,         "label/D");
}


void fill_histo(TFile* inputs, TString tname, double lambda, double mu, double nu)
{
    TTree* tree = (TTree*)inputs->Get(tname.Data());

    int nevents = tree->GetEntries();

//     cout << "---> events " << nevents << endl;

    tree->SetBranchAddress("true_phi",          &true_phi);
    tree->SetBranchAddress("true_costh",        &true_costh);
    tree->SetBranchAddress("phi",               &phi);
    tree->SetBranchAddress("costh",             &costh);

    hPar = new TH2D("hPar", "hPar", NBINS, -PI, PI, NBINS, -0.6, 0.6);
    hDet = new TH2D("hDet", "hDet", NBINS, -PI, PI, NBINS, -0.5, 0.5);

    int nfill = 0;
    int ii;

    for(ii = 0; ii < nevents && nfill < num_reco; ii++)
    {
        tree->GetEntry(ii);
        double acc = event->Uniform(0., 1.);
        if(acc > 0.5)
        {
            double weight = weight_fn(lambda, mu, nu, true_phi, true_costh);
            hPar->Fill(true_phi, true_costh, weight);
            hDet->Fill(phi, costh, weight);
            nfill++;
        }
    }

//     cout << "---> stopped @ " << ii << " event with " << nfill << " events " << endl;

    hPar->Scale(1./hPar->Integral());
    hDet->Scale(1./hDet->Integral());
}


void fill_train_tree()
{
    for(int ii = 0; ii < NBINS* NBINS; ii++)
    {
        int binx = TMath::Nint(event->Uniform(0., NBINS));
        int biny = TMath::Nint(event->Uniform(0., NBINS));

        X_par[0] = hPar->GetXaxis()->GetBinCenter(binx+1);
        X_par[1] = hPar->GetYaxis()->GetBinCenter(biny+1);
        W_par[0] = hPar->GetBinContent(binx+1, biny+1);
        W_par[1] = hPar->GetBinError(binx+1, biny+1);

        X_det[0] = hDet->GetXaxis()->GetBinCenter(binx+1);
        X_det[1] = hDet->GetYaxis()->GetBinCenter(biny+1);
        W_det[0] = hDet->GetBinContent(binx+1, biny+1);
        W_det[1] = hDet->GetBinError(binx+1, biny+1);

        if(W_par[0] > 0. && W_det[0] > 0.)
        {
            save->Fill();
//             cout << "---> filled with " << binx << " & " << biny << endl;
            break;
        }
    }
}


void fill_test_tree()
{
    for(int ii = 0; ii < NBINS; ii++)
    {
        for(int jj = 0; jj < NBINS; jj++)
        {
            X_par[0] = hPar->GetXaxis()->GetBinCenter(ii+1);
            X_par[1] = hPar->GetYaxis()->GetBinCenter(jj+1);
            W_par[0] = hPar->GetBinContent(ii+1, jj+1);
            W_par[1] = hPar->GetBinError(ii+1, jj+1);

            X_det[0] = hDet->GetXaxis()->GetBinCenter(ii+1);
            X_det[1] = hDet->GetYaxis()->GetBinCenter(jj+1);
            W_det[0] = hDet->GetBinContent(ii+1, jj+1);
            W_det[1] = hDet->GetBinError(ii+1, jj+1);

            save->Fill();
        }
    }
}


int main()
{
    // get random numbers
    for(int ii = 0; ii < num_samples; ii++)
    {
        lambda[ii] = event->Uniform(-1., 1.);
        mu[ii] = event->Uniform(-0.5, 0.5);
        nu[ii] = event->Uniform(-0.5, 0.5);
    }


    // save outputs
    TFile* inputs = TFile::Open(fname.Data(), "READ");

    TFile* outputs = new TFile("net.root", "RECREATE");

    // make X0 samples
    cout << "---> make X0 train tree " << endl;
    make_tree("X0_train_tree");

    for(int ii = 0; ii < num_samples; ii++)
    {
        thetas[0] = lambda[ii];
        thetas[1] = mu[ii];
        thetas[2] = nu[ii];

        label = 0.0;

        fill_histo(inputs, "X0_train", 0.0, 0.0, 0.0);
        fill_train_tree();
        delete hPar;
        delete hDet;

        if(ii%10000==0){cout << "---> " << ii << " events completed " << endl;}
    }

    outputs->cd();
    save->Write();
    delete save;

    // make X1 samples
    cout << "---> make X1 train tree " << endl;
    make_tree("X1_train_tree");

    for(int ii = 0; ii < num_samples; ii++)
    {
        thetas[0] = lambda[ii];
        thetas[1] = mu[ii];
        thetas[2] = nu[ii];

        label = 1.0;

        fill_histo(inputs, "X1_train", lambda[ii], mu[ii], nu[ii]);
        fill_train_tree();
        delete hPar;
        delete hDet;

        if(ii%10000==0){cout << "---> " << ii << " events completed " << endl;}
    }

    outputs->cd();
    save->Write();
    delete save;

    // make X0 secret data
    cout << "---> make X0 test tree " << endl;
    make_tree("X0_test_tree");

    thetas[0] = 0.0;
    thetas[1] = 0.0;
    thetas[2] = 0.0;

    label = 0.0;

    fill_histo(inputs, "X0_test", 0.0, 0.0, 0.0);
    fill_test_tree();

    delete hPar;
    delete hDet;

    outputs->cd();
    save->Write();
    delete save;

    // make X1 secret data
    cout << "---> make X1 test tree " << endl;
    make_tree("X1_test_tree");

    thetas[0] = lambda_secret;
    thetas[1] = mu_secret;
    thetas[2] = nu_secret;

    label = 1.0;

    fill_histo(inputs, "X1_test", lambda_secret, mu_secret, nu_secret);
    fill_test_tree();

    delete hPar;
    delete hDet;

    outputs->cd();
    save->Write();
    delete save;

    outputs->Close();

    return 0;
}
