#
#
#
import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

plt.rc("font", size=14)

def plot_hist(var: str, units: str, infile: uproot.reading.ReadOnlyDirectory)->None:

    real_name = "h{}_real".format(var)
    mc_name = "h{}_mc".format(var)
    ratio_name = "ratio_h{}".format(var)
    
    hreal = infile[real_name]
    hmc = infile[mc_name]
    hratio = infile[ratio_name]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
    axs[0].errorbar(hreal.axis().centers(), hreal.values(), hreal.errors(), hreal.axis().widths()/2., fmt="ro", label="MC data")
    axs[0].errorbar(hmc.axis().centers(), hmc.values(), hmc.errors(), hmc.axis().widths()/2., fmt="bs", label="MC data")
    axs[0].set_ylabel("Yield")
    axs[0].legend(frameon=False)

    axs[1].errorbar(hratio.axis().centers(), hratio.values(), hratio.errors(), hratio.axis().widths()/2., fmt="rs")
    axs[1].set_ylabel("Ratio")

    axs[1].set_xlabel("{} [{}]".format(var, units))
    plt.tight_layout()
    plt.savefig("imgs/compare_mc_real/before_{}.png".format(var))
    plt.close("all")

infile = uproot.open("compare.root")

var_hist = ["mass", "pT", "xF", "xT", "xB"]
var_units = ["GeV", "GeV", "a.u.", "a.u.", "a.u."]


for i in range(5):
    plot_hist(var_hist[i], var_units[i], infile)