import numpy as np
import matplotlib.pyplot as plt

def generate_blue_noise(N):
    # Generate white noise
    white_noise = np.random.randn(N)
    freqs = np.fft.rfftfreq(N)
    freqs = np.where(freqs == 0, 1e-10, freqs)
    blue_noise = np.fft.irfft(np.fft.rfft(white_noise) * (freqs))
    blue_noise = (blue_noise - np.mean(blue_noise)) / np.std(blue_noise)
    return blue_noise

def generate_violet_noise(N):
    # Generate white noise
    white_noise = np.random.randn(N)
    freqs = np.fft.rfftfreq(N)
    freqs = np.where(freqs == 0, 1e-10, freqs)
    violet_noise = np.fft.irfft(np.fft.rfft(white_noise) * (freqs)**2)
    violet_noise = (violet_noise - np.mean(violet_noise)) / np.std(violet_noise)
    return violet_noise

def generate_pink_noise(N):
    # Generate white noise
    white_noise = np.random.randn(N)
    freqs = np.fft.rfftfreq(N)
    freqs = np.where(freqs == 0, 1e-10, freqs)
    pink_noise = np.fft.irfft(np.fft.rfft(white_noise) / (freqs))
    pink_noise = (pink_noise - np.mean(pink_noise)) / np.std(pink_noise)
    return pink_noise

def generate_brown_noise(N):
    # Generate white noise
    white_noise = np.random.randn(N)
    freqs = np.fft.rfftfreq(N)
    freqs = np.where(freqs == 0, 1e-10, freqs)
    brown_noise = np.fft.irfft(np.fft.rfft(white_noise) / ((freqs)**2))
    brown_noise = (brown_noise - np.mean(brown_noise)) / np.std(brown_noise)
    return brown_noise

def generate_white_noise(N):
    # Generate white noise
    white_noise = np.random.randn(N)
    freqs = np.fft.rfftfreq(N)
    freqs = np.where(freqs == 0, 1e-10, freqs)
    brown_noise = np.fft.irfft(np.fft.rfft(white_noise) )
    brown_noise = (brown_noise - np.mean(brown_noise)) / np.std(brown_noise)
    return brown_noise


def additive_noise(x_true, Noise_1, Noise_2, n):
    Noise_train = np.concatenate((noise_level[0] * Noise_1[0:int(n / 2), ], noise_level[1] * Noise_2[0:int(n / 2), ]), axis=1)
    Noise_test = np.concatenate((noise_level[0] * Noise_1[int(n / 2):n + 1, ], noise_level[1] * Noise_2[int(n / 2):n + 1, ]), axis=1)
    X_train = x_true[0:int(n / 2)] + Noise_train
    X_test = x_true[int(n / 2):n] + Noise_test
    return X_train, X_test, Noise_train, Noise_test

Data = np.load('Noisefreedata.npz')
Time = Data['Time']
Voltage = Data['Voltage']
AdaptPar = Data['AdaptPar']
Current = Data['Current']
n = len(Time)
x_true = np.concatenate((Voltage, AdaptPar),axis=1)
x_true_rms1 = ((x_true[:,0]**2).mean())**0.5
x_true_rms2 = ((x_true[:,1]**2).mean())**0.5
noise_level = 0.10*np.array([x_true_rms1, x_true_rms2])
Time_train = Time[0:int(n/2)]
Time_test = Time[int(n/2)+1:n]
Time_total = np.concatenate((Time_train,Time_test),axis=0)
y_train = x_true[0:int(n/2)]
y_test = x_true[int(n/2):n]

# Generate blue noise
blue_noise_1 = generate_blue_noise(n)
blue_noise_2 = generate_blue_noise(n)
blue_noise_1 = (blue_noise_1.reshape(-1,1) - np.average(blue_noise_1)) / np.std(blue_noise_1)
blue_noise_2 = (blue_noise_2.reshape(-1,1) - np.average(blue_noise_2)) / np.std(blue_noise_2)

# Generate violet noise
violet_noise_1 = generate_violet_noise(n)
violet_noise_2 = generate_violet_noise(n)
violet_noise_1 = (violet_noise_1.reshape(-1,1) - np.average(violet_noise_1)) / np.std(violet_noise_1)
violet_noise_2 = (violet_noise_2.reshape(-1,1) - np.average(violet_noise_2)) / np.std(violet_noise_2)

# Generate pink noise
pink_noise_1 = generate_pink_noise(n)
pink_noise_2 = generate_pink_noise(n)
pink_noise_1 = (pink_noise_1.reshape(-1,1) - np.average(pink_noise_1)) / np.std(pink_noise_1)
pink_noise_2 = (pink_noise_2.reshape(-1,1) - np.average(pink_noise_2)) / np.std(pink_noise_2)

# Generate brown noise
brown_noise_1 = generate_brown_noise(n)
brown_noise_2 = generate_brown_noise(n)
brown_noise_1 = (brown_noise_1.reshape(-1,1) - np.average(brown_noise_1)) / np.std(brown_noise_1)
brown_noise_2 = (brown_noise_2.reshape(-1,1) - np.average(brown_noise_2)) / np.std(brown_noise_2)

# Generate white noise
white_noise_1 = generate_white_noise(n)
white_noise_2 = generate_white_noise(n)
white_noise_1 = (white_noise_1.reshape(-1,1) - np.average(white_noise_1)) / np.std(white_noise_1)
white_noise_2 = (white_noise_2.reshape(-1,1) - np.average(white_noise_2)) / np.std(white_noise_2)

# Plot noise processes
plt.figure(figsize = (5,4))
plt.style.use('./Styles/AIPStyles.mplstyle')
NFFT = 512  # Length of each segment
plt.psd(blue_noise_1[:,0], NFFT=NFFT, noverlap=NFFT//2, Fs=100000, color='blue', label = 'Blue Noise')
plt.psd(violet_noise_1[:,0], NFFT=NFFT, noverlap=NFFT//2, Fs=100000, color='violet', label = 'Violet Noise')
plt.psd(pink_noise_1[:,0], NFFT=NFFT, noverlap=NFFT//2, Fs=100000, color='pink', label = 'Pink Noise')
plt.psd(brown_noise_1[:,0], NFFT=NFFT, noverlap=NFFT//2, Fs=100000, color='Brown', label = 'Brown Noise')
plt.psd(white_noise_1[:,0], NFFT=NFFT, noverlap=NFFT//2, Fs=100000, color='black', label = 'White Noise')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD (dB/Hz)')
plt.xlim([0,50000])
plt.grid(False)
plt.legend(loc='lower right')
plt.ylim([-162,-22])
plt.tight_layout()
plt.show()

# Add blue noise to the observations
X_train, X_test, Noise_train_blue, Noise_test_blue = additive_noise(x_true, blue_noise_1, blue_noise_2, n)
np.savetxt('.\Dataset\AINF_Dataset10p_Blue\X_true.txt', x_true, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Blue\X_train.txt', X_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Blue\y_train.txt', y_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Blue\Time_train.txt', Time_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Blue\X_test.txt', X_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Blue\y_test.txt', y_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Blue\Time_test.txt', Time_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Blue\Time.txt', Time, delimiter=' ')

# Add violet noise to the observations
X_train, X_test, Noise_train_violet, Noise_test_violet = additive_noise(x_true, violet_noise_1, violet_noise_2, n)
np.savetxt('.\Dataset\AINF_Dataset10p_Violet\X_true.txt', x_true, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Violet\X_train.txt', X_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Violet\y_train.txt', y_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Violet\Time_train.txt', Time_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Violet\X_test.txt', X_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Violet\y_test.txt', y_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Violet\Time_test.txt', Time_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Violet\Time.txt', Time, delimiter=' ')

# Add pink noise to the observations
X_train, X_test, Noise_train_pink, Noise_test_pink = additive_noise(x_true, pink_noise_1, pink_noise_2, n)
np.savetxt('.\Dataset\AINF_Dataset10p_Pink\X_true.txt', x_true, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Pink\X_train.txt', X_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Pink\y_train.txt', y_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Pink\Time_train.txt', Time_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Pink\X_test.txt', X_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Pink\y_test.txt', y_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Pink\Time_test.txt', Time_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Pink\Time.txt', Time, delimiter=' ')

# Add brown noise to the observations
X_train, X_test, Noise_train_brown, Noise_test_brown = additive_noise(x_true, brown_noise_1, brown_noise_2, n)
np.savetxt('.\Dataset\AINF_Dataset10p_Brown\X_true.txt', x_true, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Brown\X_train.txt', X_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Brown\y_train.txt', y_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Brown\Time_train.txt', Time_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Brown\X_test.txt', X_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Brown\y_test.txt', y_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Brown\Time_test.txt', Time_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_Brown\Time.txt', Time, delimiter=' ')

# Add white noise to the observations
X_train, X_test, Noise_train_white, Noise_test_white = additive_noise(x_true, white_noise_1, white_noise_2, n)
np.savetxt('.\Dataset\AINF_Dataset10p_GWN\X_true.txt', x_true, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_GWN\X_train.txt', X_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_GWN\y_train.txt', y_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_GWN\Time_train.txt', Time_train, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_GWN\X_test.txt', X_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_GWN\y_test.txt', y_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_GWN\Time_test.txt', Time_test, delimiter=' ')
np.savetxt('.\Dataset\AINF_Dataset10p_GWN\Time.txt', Time, delimiter=' ')

# X_Total = np.concatenate((X_train, X_test))
# Noise_Total = np.concatenate((Noise_train, Noise_test))

plt.figure(figsize = (15,8))
plt.style.use('./Styles/AIPStyles.mplstyle')
plt.subplot(5,1,1)
plt.plot(Noise_train_blue[:,0], color='blue')
plt.subplot(5,1,2)
plt.plot(Noise_train_violet[:,0], color='violet')
plt.subplot(5,1,3)
plt.plot(Noise_train_pink[:,0], color='pink')
plt.subplot(5,1,4)
plt.plot(Noise_train_brown[:,0], color='brown')
plt.subplot(5,1,5)
plt.plot(Noise_train_white[:,0], color='black')
plt.ylabel(r"$I(t) \ A$")
plt.xlabel('Time (s)')
plt.legend(loc='lower right')
plt.show()
