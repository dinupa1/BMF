#
# D1 occupancy cut
# 26March2023
# dinupa3@gmail.com
#

import uproot
import numpy as np
import awkward as ak

import ROOT

PI = ROOT.TMath.Pi()

# some custom style
# ROOT.gStyle.SetHistFillColor(422)
ROOT.gStyle.SetHistLineStyle(2)
ROOT.gStyle.SetHistLineWidth(2)
# ROOT.gStyle.SetStatFontSize(0.1)
ROOT.gStyle.SetStatY(0.85)
ROOT.gStyle.SetStatX(0.85)
ROOT.gStyle.SetStatW(0.3)
ROOT.gStyle.SetStatH(0.3)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetStatBorderSize(0)
ROOT.gStyle.SetStatFont(42)
ROOT.gStyle.SetCanvasDefH(800)
ROOT.gStyle.SetCanvasDefW(800)
ROOT.gStyle.SetLegendBorderSize(0)
ROOT.gStyle.SetHistMinimumZero(1)

tree = uproot.open("../data.root:save")
events = tree.arrays(["occuD1", "true_mass", "true_pT", "true_x1", "true_x2", "true_xF", "true_phi", "true_costh",
                      "mass", "pT", "x1", "x2", "xF", "phi", "costh"])

can = ROOT.TCanvas()

occuD1H = ROOT.TH1D("occuD1", "; occuD1 [a.u.]; counts [a.u.]", 50, 0., 500.)
[occuD1H.Fill(m) for m in events.occuD1]

occuD1H.Draw()
can.SaveAs("imgs/occuD1.png")

massH = ROOT.TH1D("mass", "; mass [GeV/c^2]; counts [a.u.]", 50, 3., 8.)
[massH.Fill(m) for m in events.mass]

massD1H = ROOT.TH1D("massD1", "; mass [GeV/c^2]; counts [a.u.]", 50, 3., 8.)
[massD1H.Fill(m) for m in events.mass[events.occuD1 < 200.]]

L1 = ROOT.TLegend(0.5, 0.6, 0.9, 0.8)
L1.AddEntry(massH, "without D1 occu. cut")
L1.AddEntry(massD1H, "with D1 occu. cut")

massH.Draw()
massD1H.SetLineColor(632)
massD1H.Draw("SAME")
L1.Draw()
can.SaveAs("imgs/mass.png")

pTH = ROOT.TH1D("pT", "; pT [GeV/c]; counts [a.u.]", 50, 0., 2.5)
[pTH.Fill(m) for m in events.pT]

pTD1H = ROOT.TH1D("pTD1", "; pT [GeV/c]; counts [a.u.]", 50, 0., 2.5)
[pTD1H.Fill(m) for m in events.pT[events.occuD1 < 200.]]

L2 = ROOT.TLegend(0.5, 0.6, 0.9, 0.8)
L2.AddEntry(pTH, "without D1 occu. cut")
L2.AddEntry(pTD1H, "with D1 occu. cut")

pTH.Draw()
pTD1H.SetLineColor(632)
pTD1H.Draw("SAME")
L2.Draw()
can.SaveAs("imgs/pT.png")

x1H = ROOT.TH1D("x1", "; x1; counts [a.u.]", 50, 0.3, 1.)
[x1H.Fill(m) for m in events.x1]

x1D1H = ROOT.TH1D("x1D1", "; x1; counts [a.u.]", 50, 0.3, 1.)
[x1D1H.Fill(m) for m in events.x1[events.occuD1 < 200.]]

L3 = ROOT.TLegend(0.5, 0.6, 0.9, 0.8)
L3.AddEntry(x1H, "without D1 occu. cut")
L3.AddEntry(x1D1H, "with D1 occu. cut")

x1H.Draw()
x1D1H.SetLineColor(632)
x1D1H.Draw("SAME")
L3.Draw()
can.SaveAs("imgs/x1.png")

x2H = ROOT.TH1D("x2", "; x2; counts [a.u.]", 50, 0.1, 0.5)
[x2H.Fill(m) for m in events.x2]

x2D1H = ROOT.TH1D("x2D1", "; x2; counts [a.u.]", 50, 0.1, 0.5)
[x2D1H.Fill(m) for m in events.x2[events.occuD1 < 200.]]

L4 = ROOT.TLegend(0.5, 0.6, 0.9, 0.8)
L4.AddEntry(x2H, "without D1 occu. cut")
L4.AddEntry(x2D1H, "with D1 occu. cut")

x2H.Draw()
x2D1H.SetLineColor(632)
x2D1H.Draw("SAME")
L4.Draw()
can.SaveAs("imgs/x2.png")

xFH = ROOT.TH1D("xF", "; xF; counts [a.u.]", 50, 0., 1.)
[xFH.Fill(m) for m in events.xF]

xFD1H = ROOT.TH1D("xFD1", "; xF; counts [a.u.]", 50, 0., 1.)
[xFD1H.Fill(m) for m in events.xF[events.occuD1 < 200.]]

L5 = ROOT.TLegend(0.5, 0.6, 0.9, 0.8)
L5.AddEntry(xFH, "without D1 occu. cut")
L5.AddEntry(xFD1H, "with D1 occu. cut")

xFH.Draw()
xFD1H.SetLineColor(632)
xFD1H.Draw("SAME")
L5.Draw()
can.SaveAs("imgs/xF.png")


phiH = ROOT.TH1D("phi", "; phi [rad]; counts [a.u.]", 50, -PI, PI)
[phiH.Fill(m) for m in events.phi]

phiD1H = ROOT.TH1D("phiD1", "; phi [rad]; counts [a.u.]", 50, -PI, PI)
[phiD1H.Fill(m) for m in events.phi[events.occuD1 < 200.]]

L6 = ROOT.TLegend(0.5, 0.2, 0.9, 0.4)
L6.AddEntry(phiH, "without D1 occu. cut")
L6.AddEntry(phiD1H, "with D1 occu. cut")

phiH.Draw()
phiD1H.SetLineColor(632)
phiD1H.Draw("SAME")
L6.Draw()
can.SaveAs("imgs/phi.png")

costhH = ROOT.TH1D("costh", "; costh; counts [a.u.]", 50, -0.6, 0.6)
[costhH.Fill(m) for m in events.costh]

costhD1H = ROOT.TH1D("costhD1", "; costh; counts [a.u.]", 50, -0.6, 0.6)
[costhD1H.Fill(m) for m in events.costh[events.occuD1 < 200.]]

L7 = ROOT.TLegend(0.5, 0.2, 0.9, 0.4)
L7.AddEntry(pTH, "without D1 occu. cut")
L7.AddEntry(pTD1H, "with D1 occu. cut")

costhH.Draw()
costhD1H.SetLineColor(632)
costhD1H.Draw("SAME")
L7.Draw()
can.SaveAs("imgs/costh.png")


# ------------------------------------------
events1 = events[events.occuD1 < 200.]

mass2D = ROOT.TH2D("mass2D", "; true [GeV/c^2]; reco [GeV/c^2]", 50, 3., 8., 50, 3., 8.)
[mass2D.Fill(m, n) for m, n in zip(events1.true_mass, events1.mass)]

mass2D.Draw("COLZ")
can.SaveAs("imgs/mass2D.png")

pT2D = ROOT.TH2D("pT2D", "; true [GeV/c]; reco [GeV/c]", 50, 0., 2.5, 50, 0., 2.5)
[pT2D.Fill(m, n) for m, n in zip(events1.true_pT, events1.pT)]

pT2D.Draw("COLZ")
can.SaveAs("imgs/pT2D.png")

x12D = ROOT.TH2D("x12D", "; true; reco", 50, 0.3, 1., 50, 0.3, 1.)
[x12D.Fill(m, n) for m, n in zip(events1.true_x1, events1.x1)]

x12D.Draw("COLZ")
can.SaveAs("imgs/x12D.png")

x22D = ROOT.TH2D("x22D", "; true; reco", 50, 0.1, 0.5, 50, 0.1, 0.5)
[x22D.Fill(m, n) for m, n in zip(events1.true_x2, events1.x2)]

x22D.Draw("COLZ")
can.SaveAs("imgs/x22D.png")

xF2D = ROOT.TH2D("xF2D", "; true; reco", 50, 0., 1., 50, 0., 1.)
[xF2D.Fill(m, n) for m, n in zip(events1.true_xF, events1.xF)]

xF2D.Draw("COLZ")
can.SaveAs("imgs/xF2D.png")

phi2D = ROOT.TH2D("phi2D", "; true [rad]; reco [rad]", 50, -PI, PI, 50, -PI, PI)
[phi2D.Fill(m, n) for m, n in zip(events1.true_phi, events1.phi)]

phi2D.Draw("COLZ")
can.SaveAs("imgs/phi2D.png")

costh2D = ROOT.TH2D("costh2D", "; true; reco", 50, -0.6, 0.6, 50, -0.6, 0.6)
[costh2D.Fill(m, n) for m, n in zip(events1.true_costh, events1.costh)]

costh2D.Draw("COLZ")
can.SaveAs("imgs/costh2D.png")