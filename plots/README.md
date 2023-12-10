# Plot the predictions

Use the following commands to create plots.

```
mkdir build
cd build
root -b -e 'gSystem->CompileMacro("../Plots.cc", "kfgO", "plots");' -q

cd ..
mkdir imgs
root -b -e 'gSystem->Load("build/plots.so"); Plots();' -q
```

Plots are saved in the `imgs` directory.