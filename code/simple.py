# ##################################### #
# simple example for BM function        #
# dinupa3@gmail.com                     #
# 13-feb-2023                           #
# ##################################### #


import numpy as np
import matplotlib.pyplot as plt

p_unp2 = 0.25
p_bm2 = 0.161

Q_T = np.linspace(0.0, 5.0, 20)
Q_T2 = Q_T* Q_T
exp_p_bm = Q_T2/(2* p_bm2)
exp_p_unp = Q_T2/(2* p_unp2)
nu = Q_T2* np.exp(-exp_p_bm)/np.exp(-exp_p_unp)

plt.subplots()
plt.plot(Q_T, nu, "o")
plt.xlabel("Q_T")
plt.ylabel("nu")
plt.savefig("imgs/simple.png")
plt.show()
