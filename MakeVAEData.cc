//
// dinupa3@gmail.com
//

#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TMath.h>
#include <TRandom.h>
#include <TCanvas.h>
#include <iostream>

using namespace std;

double pi = TMath::Pi();


double weight_fn(double lambda, double mu, double nu, double phi, double costh)
{
    double weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
    return weight/(1. + costh* costh);
}

class VAEData{
    // Inputs
    TTree* tree;
    int n_events;
    double mass, pT, xF, phi, costh, true_phi, true_costh;

    // Outputs
    TFile* outfile;
    TTree* train_tree;
    TTree* secret_tree;
    double counts[144];
    double theta[3];
    double label[2];
    TRandom3* r_event = new TRandom3(1);
    TRandom3* r_lambda = new TRandom3(2);
    TRandom3* r_mu = new TRandom3(3);
    TRandom3* r_nu = new TRandom3(4);

    public:
        VAEData();
        virtual ~VAEData(){;}
        int Init(TString outfile_name);
        TH2D* MakeHist(double lambda, double mu, double nu, int event0, int event1);
        int FillTree(TTree* tree, TH2D* hist, double lambda, double mu, double nu, double lable0, double label1);
        int FillTrainTree(int events);
        int FillSecretTree(int events, double lambda, double mu, double nu);
        int End();
};


VAEData::VAEData()
{;}

int VAEData::Init(TString outfile_name)
{
    auto file = TFile::Open("data.root", "READ");
    tree = (TTree*)file->Get("save");
    n_events = tree->GetEntries();

    cout << "===> total # of events = " << n_events << endl;

    tree->SetBranchAddress("mass", &mass);
    tree->SetBranchAddress("pT", &pT);
    tree->SetBranchAddress("xF", &xF);
    tree->SetBranchAddress("pT", &pT);
    tree->SetBranchAddress("phi", &phi);
    tree->SetBranchAddress("costh", &costh);
    tree->SetBranchAddress("true_phi", &true_phi);
    tree->SetBranchAddress("true_costh", &true_costh);

    tree->LoadBaskets(99999999999);

    outfile = new TFile(outfile_name.Data(), "RECREATE");

    train_tree = new TTree("train_tree", "histograms for VAE training");

    train_tree->Branch("counts", counts, "counts[144]/D");
    train_tree->Branch("theta", theta, "theta[3]/D");
    train_tree->Branch("label", label, "label[2]/D");

    secret_tree = new TTree("secret_tree", "histograms with secret theta");

    secret_tree->Branch("counts", counts, "counts[144]/D");
    secret_tree->Branch("theta", theta, "theta[3]/D");
    secret_tree->Branch("label", label, "label[2]/D");

    return 0;
}

TH2D* VAEData::MakeHist(double lambda, double mu, double nu, int event0, int event1)
{
    auto hist = new TH2D("hist", "; phi [rand]; costh [a.u.]", 12, -pi, pi, 12, -0.5, 0.5);

    for(int i = 0; i < 15000; i++)
    {
        double event = r_event->Uniform(event0, event1);
        tree->GetEntry(TMath::Nint(event));
        // cout << "event = " << TMath::Nint(event) << " phi = " << phi << " costh = " << costh << " true phi " << true_phi << " true costh = " << true_costh << endl;
        if(4.5 < mass){hist->Fill(phi, costh, weight_fn(lambda, mu, nu, true_phi, true_costh));}
    }

    hist->Scale(1./hist->Integral());

    // auto can = new TCanvas();
    // hist->Draw("COLZ");
    // can->SaveAs("imgs/scrach.png");

    return hist;
}


int VAEData::FillTree(TTree* tree, TH2D* hist, double lambda, double mu, double nu, double label0, double label1)
{
    for(int i = 0; i < 12; i++)
    {
        for(int j = 0; j < 12; j++)
        {
            counts[12*i + j] = hist->GetBinContent(i+1, j+1);
            //cout << "entry = " << 12*i + j << " count = " << hist->GetBinContent(i+1, j+1) << endl;
        }
    }

    theta[0] = lambda;
    theta[1] = mu;
    theta[2] = nu;

    label[0] = label0;
    label[1] = label1;

    tree->Fill();

    return 0;
}

int VAEData::FillTrainTree(int events)
{
    for(int i = 0; i < events; i++)
    {
        double lambda = r_lambda->Uniform(-1.0, 1.0);
        double mu = r_mu->Uniform(-0.5, 0.5);
        double nu = r_nu->Uniform(-0.5, 0.5);

        auto hist0 = (TH2D*)MakeHist(0.0, 0.0, 0.0, 0, n_events/3);

        FillTree(train_tree, hist0, lambda, mu, nu, 1.0, 0.0);

        delete hist0;

        auto hist1 = (TH2D*)MakeHist(lambda, mu, nu, n_events/3, 2*n_events/3);

        FillTree(train_tree, hist1, lambda, mu, nu, 0.0, 1.0);

        delete hist1;

        if(i%10000 == 0)
        {
            TString outputs = Form("===> creating %d histograms with lambda = %.3f, mu = %.3f, nu = %.3f", i, lambda, mu, nu);
            cout << outputs.Data() << endl;
        }
    }

    return 0;
}

int VAEData::FillSecretTree(int events, double lambda, double mu, double nu)
{

    for(int i = 0; i < events; i++)
    {
        auto hist1_secret = (TH2D*)MakeHist(lambda, mu, nu, 2*n_events/3, n_events);

        FillTree(secret_tree, hist1_secret, lambda, mu, nu, 0.0, 1.0);

        delete hist1_secret;

        if(i%10000 == 0)
        {
            TString outputs = Form("===> creating %d histograms with lambda = %.3f, mu = %.3f, nu = %.3f", i, lambda, mu, nu);
            cout << outputs.Data() << endl;
        }
    }

    return 0;
}


int VAEData::End()
{
    train_tree->Write();
    secret_tree->Write();
    outfile->Close();
    return 0;
}


void MakeVAEData()
{
    auto vae_data = new VAEData();
    vae_data->Init("vae_data.root");
    vae_data->FillTrainTree(100000);
    vae_data->FillSecretTree(30000, 0.8, 0.1, 0.2);
    vae_data->End();
}
