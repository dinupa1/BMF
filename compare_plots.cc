#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TString.h>
#include <TCanvas.h>
#include <iostream>


void normalize(TH1D* hist, TH1D* hist_mc, TH1D* hist_mc2)
{
    hist->Scale(1./hist->Integral());
    hist_mc->Scale(1./hist_mc->Integral());
    hist_mc2->Scale(1./hist_mc2->Integral());
}

void plots(TH1D* h1, TH1D* h2, TString fname)
{
    auto can = new TCanvas(fname.Data(), fname.Data(), 800, 800);

    double xmax = 1.5* h1->GetMaximum();
    h1->SetMaximum(xmax);

    h1->SetFillColorAlpha(kTeal+1, 0.3);
    h2->SetMarkerColor(kAzure+1);
    h2->SetMarkerStyle(20);

    h1->Draw("HIST");
    h2->Draw("E1 SAME");

    TString save_name = Form("imgs/%s.png", fname.Data());

    can->SaveAs(save_name.Data());
}


void ratio_plots(TH1D* hist, TH1D* hist_mc, TString var, TString units, TString fname)
{
    hist_mc->Divide(hist);

    TString hname = Form("ratio_%s_%s", var.Data(), fname.Data());
    TString htitle = Form("; %s [%s]; #frac{MC}{Real}", var.Data(), units.Data());

    hist_mc->SetNameTitle(hname.Data(), htitle.Data());

    hist_mc->SetMarkerColor(kAzure+1);
    hist_mc->SetMarkerStyle(20);

    TString cname = Form("can_%s", hname.Data());

    auto c = new TCanvas(cname.Data(), cname.Data(), 800, 800);
    hist_mc->Draw("E1");

    TString pname = Form("imgs/%s_ratio_%s", fname.Data(), var.Data());
    c->SaveAs(pname.Data());
}


void compare_plots()
{
    TH1::SetDefaultSumw2();

    auto infile = TFile::Open("../e906-LH2-data/output.root", "READ");
    auto tree = (TTree*)infile->Get("tree");
    auto tree_mc = (TTree*)infile->Get("tree_mc");

    int nevents = tree->GetEntries();
    int nevents_mc = tree->GetEntries();

    std::cout << "---> mc events " << nevents_mc << " real events " <<  nevents <<std::endl;

    double mass, pT, xT, xB, xF, weight;
    double mass_mc, pT_mc, xT_mc, xB_mc, xF_mc, weight_mc, weight2_mc;

    tree->SetBranchAddress("mass", &mass);
    tree->SetBranchAddress("pT", &pT);
    tree->SetBranchAddress("xT", &xT);
    tree->SetBranchAddress("xB", &xB);
    tree->SetBranchAddress("xF", &xF);
    tree->SetBranchAddress("weight", &weight);

    tree_mc->SetBranchAddress("mass", &mass_mc);
    tree_mc->SetBranchAddress("pT", &pT_mc);
    tree_mc->SetBranchAddress("xT", &xT_mc);
    tree_mc->SetBranchAddress("xB", &xB_mc);
    tree_mc->SetBranchAddress("xF", &xF_mc);
    tree_mc->SetBranchAddress("weight", &weight_mc);
    tree_mc->SetBranchAddress("weight2", &weight2_mc);

    auto hmass = new TH1D("hmass", "; mass [GeV]; counts", 20, 4.5, 9.0);
    auto hpT = new TH1D("hpT", "; pT [GeV]; counts", 20, 0.0, 3.0);
    auto hxT = new TH1D("hxT", "; xT; counts", 20, 0.1, 0.6);
    auto hxB = new TH1D("hxB", "; xB; counts", 20, 0.3, 1.0);
    auto hxF = new TH1D("hxF", "; xF; counts", 20, -0.1, 1.0);

    auto hmass_mc = new TH1D("hmass_mc", "; mass [GeV]; counts", 20, 4.5, 9.0);
    auto hpT_mc = new TH1D("hpT_mc", "; pT [GeV]; counts", 20, 0.0, 3.0);
    auto hxT_mc = new TH1D("hxT_mc", "; xT; counts", 20, 0.1, 0.6);
    auto hxB_mc = new TH1D("hxB_mc", "; xB; counts", 20, 0.3, 1.0);
    auto hxF_mc = new TH1D("hxF_mc", "; xF; counts", 20, -0.1, 1.0);

    auto hmass_mc2 = new TH1D("hmass_mc2", "; mass [GeV]; counts", 20, 4.5, 9.0);
    auto hpT_mc2 = new TH1D("hpT_mc2", "; pT [GeV]; counts", 20, 0.0, 3.0);
    auto hxT_mc2 = new TH1D("hxT_mc2", "; xT; counts", 20, 0.1, 0.6);
    auto hxB_mc2 = new TH1D("hxB_mc2", "; xB; counts", 20, 0.3, 1.0);
    auto hxF_mc2 = new TH1D("hxF_mc2", "; xF; counts", 20, -0.1, 1.0);


    for(int ii = 0; ii < nevents; ii++)
    {
        tree->GetEntry(ii);
        tree_mc->GetEntry(ii);

        hmass->Fill(mass, weight);
        hpT->Fill(pT, weight);
        hxB->Fill(xB, weight);
        hxT->Fill(xT, weight);
        hxF->Fill(xF, weight);

        hmass_mc->Fill(mass, weight_mc);
        hpT_mc->Fill(pT, weight_mc);
        hxB_mc->Fill(xB, weight_mc);
        hxT_mc->Fill(xT, weight_mc);
        hxF_mc->Fill(xF, weight_mc);

        hmass_mc2->Fill(mass, weight2_mc);
        hpT_mc2->Fill(pT, weight2_mc);
        hxB_mc2->Fill(xB, weight2_mc);
        hxT_mc2->Fill(xT, weight2_mc);
        hxF_mc2->Fill(xF, weight2_mc);
    }

    normalize(hmass, hmass_mc, hmass_mc2);
    normalize(hpT, hpT_mc, hpT_mc2);
    normalize(hxB, hxB_mc, hxB_mc2);
    normalize(hxT, hxT_mc, hxT_mc2);
    normalize(hxF, hxF_mc, hxF_mc2);

    plots(hmass, hmass_mc, "mass");
    plots(hpT, hpT_mc, "pT");
    plots(hxB, hxB_mc, "xB");
    plots(hxT, hxT_mc, "xT");
    plots(hxF, hxF_mc, "xF");

    ratio_plots(hmass, hmass_mc, "mass", "GeV", "before");
    ratio_plots(hpT, hpT_mc, "pT", "GeV", "before");
    ratio_plots(hxB, hxB_mc, "xB", " ", "before");
    ratio_plots(hxT, hxT_mc, "xT", " ", "before");
    ratio_plots(hxF, hxF_mc, "xF", " ", "before");

    ratio_plots(hmass, hmass_mc2, "mass", "GeV", "after");
    ratio_plots(hpT, hpT_mc2, "pT", "GeV", "after");
    ratio_plots(hxB, hxB_mc2, "xB", " ", "after");
    ratio_plots(hxT, hxT_mc2, "xT", " ", "after");
    ratio_plots(hxF, hxF_mc2, "xF", " ", "after");

}
