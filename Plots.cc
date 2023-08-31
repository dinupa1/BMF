//
// dinupa3@gmail.com
// 08-30-2023
//

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <iostream>

using namespace std;

void Plots()
{
    gStyle->SetOptFit();
    auto file = TFile::Open("results.root", "READ");
    auto tree = (TTree*)file->Get("opt_params");
    int iterations = tree->GetEntries();
    //cout << "Iterations : " << iterations << endl;

    // target values
    double targt_lambda = 0.8, target_mu = -0.1, target_nu = 0.2;

    // opt params
    double opt_lambda, opt_mu, opt_nu;

    tree->SetBranchAddress("lambda", &opt_lambda);
    tree->SetBranchAddress("mu", &opt_mu);
    tree->SetBranchAddress("nu", &opt_nu);

    // make hist
    auto lambda_hist = new TH1D("lambda_hist", "#lambda_{injected} = 0.8; #lambda [a.u.]; counts [a.u.]", 20, 0.0, 2.0);
    auto mu_hist = new TH1D("mu_hist", "#mu_{injected} = -0.1; #mu [a.u.]; counts [a.u.]", 20, -1.0, 1.0);
    auto nu_hist = new TH1D("nu_hist", "#nu_{injected} = 0.2; #nu [a.u.]; counts [a.u.]", 20, -1.0, 1.0);

    for(int i = 0; i < iterations; i++)
    {
        tree->GetEntry(i);
        lambda_hist->Fill(opt_lambda);
        mu_hist->Fill(opt_mu);
        nu_hist->Fill(opt_nu);

    }

    auto can = new TCanvas();

    lambda_hist->Fit("gaus");
    lambda_hist->Draw();
    can->SaveAs("imgs/opt_lambda_4.png");


    mu_hist->Fit("gaus");
    mu_hist->Draw();
    can->SaveAs("imgs/opt_mu_4.png");

    nu_hist->Fit("gaus");
    nu_hist->Draw();
    can->SaveAs("imgs/opt_nu_4.png");

}
