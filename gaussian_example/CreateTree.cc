#include<TFile.h>
#include<TTree.h>
#include<TH1D.h>
#include<TMath.h>
#include<TString.h>
#include<iostream>

using namespace std;

int num_events = 1000000;
const int num_samples = 100000;

int num_bins = 30;
double xmin = -6.;
double xmax = 6.;

double mu0 = 0.0;
double sigma = 1.;

double x, y, weight, x_err, theta;

double mu_secret = -1.2;

TH1D* MakeHisto(TString hist_name, double mu, TRandom3* event)
{
    TH1D* hist = new TH1D(hist_name.Data(), "; x [a.u.]; normalized to unity", num_bins, xmin, xmax);

    for(int i = 0; i < num_events; i++)
    {
        hist->Fill(event->Gaus(mu, sigma));
    }

    hist->Scale(1./hist->Integral());
    return hist;
}

TTree* MakeTree(TString tree_name)
{
    TTree* tree = new TTree(tree_name.Data(), tree_name.Data());
    tree->Branch("x",       &x,         "x/D");
    tree->Branch("y",       &y,         "y/D");
    tree->Branch("weight",  &weight,    "weight/D");
    tree->Branch("x_err",   &x_err,     "x_err/D");
    tree->Branch("theta",   &theta,     "theta/D");
    return tree;
}

void FillTrainTree(TTree* tree, TH1D* hist, double label, double mu, TRandom3* event)
{
    for(int i = 0; i < num_bins; i++)
    {
        double bin_num = event->Uniform(0., 30.);
        weight = hist->GetBinContent(TMath::Nint(bin_num) + 1);
        x_err = hist->GetBinError(TMath::Nint(bin_num) + 1);
        x = hist->GetBinCenter(TMath::Nint(bin_num) + 1);
        y = label;
        theta = mu;
        if(weight > 0.)
        {
            tree->Fill();
            break;
        }
    }

}

void FillTestTree(TTree* tree, TH1D* hist, double label, double mu)
{
    for(int i = 0; i < num_bins; i++)
    {
        weight = hist->GetBinContent(i + 1);
        x_err = hist->GetBinError(i + 1);
        x = hist->GetBinCenter(i + 1);
        y = label;
        theta = mu;
        tree->Fill();
    }
}

void CreateTree()
{
    TRandom3* event = new TRandom3();

    TFile* outFile = new TFile("gauss_data.root", "recreate");

    TTree* X0_train_tree = (TTree*)MakeTree("X0_train_tree");
    TTree* X1_train_tree = (TTree*)MakeTree("X1_train_tree");
    TTree* X0_test_tree = (TTree*)MakeTree("X0_test_tree");
    TTree* X1_test_tree = (TTree*)MakeTree("X1_test_tree");

    double mu_values[num_samples];

    for(int i = 0; i < num_samples; i++)
    {
        mu_values[i] = event->Uniform(-2., 2.);
    }

    for(int i = 0; i < num_samples; i++)
    {
        TString X0_train_hist_name = Form("X0_train_hist_%d", i);
        TH1D* X0_train_hist = (TH1D*)MakeHisto(X0_train_hist_name, mu0, event);
        FillTrainTree(X0_train_tree, X0_train_hist, 0.0, mu_values[i], event);
        delete X0_train_hist;

        TString X1_train_hist_name = Form("X1_train_hist_%d", i);
        TH1D* X1_train_hist = (TH1D*)MakeHisto(X1_train_hist_name, mu_values[i], event);
        FillTrainTree(X1_train_tree, X1_train_hist, 1.0, mu_values[i], event);
        delete X1_train_hist;

        if(i%10000 == 0){cout << "---> starting event = " << i << endl;}
    }

    TH1D* X0_test_hist = (TH1D*)MakeHisto("X0_test_hist", mu0, event);
    FillTestTree(X0_test_tree, X0_test_hist, 0.0, mu0);
    delete X0_test_hist;

    TH1D* X1_test_hist = (TH1D*)MakeHisto("X1_test_hist", mu_secret, event);
    FillTestTree(X1_test_tree, X1_test_hist, 1.0, mu_secret);
    delete X1_test_hist;

    X0_train_tree->Write();
    X1_train_tree->Write();
    X0_test_tree->Write();
    X1_test_tree->Write();
    outFile->Close();
}
