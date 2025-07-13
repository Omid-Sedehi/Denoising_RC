import brian2 as b2
import numpy as np
from neurodynex3.adex_model import AdEx
from neurodynex3.tools import input_factory
import matplotlib.pyplot as plt

current = input_factory.get_step_current(10, 400, 1. * b2.ms, 65.0 * b2.pA)
state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=400 * b2.ms,
                                                         v_spike=-10 * b2.mV, v_rheobase=-55* b2.mV,
                                                         tau_m=5.0*b2.ms, tau_w=100*b2.ms,
                                                         a=-0.5*b2.nS,b=7.0*b2.pA, delta_T=2*b2.mV)
Time = np.arange(0,400*0.001,1e-5).reshape(-1,1)
Voltage = np.array(state_monitor.v.T)
AdaptPar = np.array(state_monitor.w.T)
Current = np.concatenate((np.zeros(1000),np.ones(39000)*6.5e-11))
np.savez('Noisefreedata.npz', Time = Time, Voltage = Voltage, AdaptPar = AdaptPar, Current = Current)

# Plot the results For Pruning
plt.figure(figsize=(5, 2.5))
plt.style.use('./Styles/AIPStyles.mplstyle')
plt.plot(Time, 1e12 * Current, linestyle='-', color='seagreen', linewidth=1)
plt.ylabel(r'$I \ (pA)$', labelpad=-2)
plt.xlabel(r'$t \ (s)$', labelpad=-2)
plt.xlim([0, 0.4])
# plt.ylim([1000 * -0.090, 1000 * 0.010])
plt.legend(loc='upper right', ncol=3)
plt.text(0.02, 0.95, 'b)', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
