'''
@Modified by Paul Cho; 10th, Nov, 2020

For NTHU DSP Lab 2022 Autumn
'''

from turtle import pencolor
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from librosa.filters import mel as librosa_mel_fn
from scipy.fftpack import dct

from Lab1_functions_student import pre_emphasis, STFT, mel2hz, hz2mel, get_filter_banks
from Lab2_stft2audio_student import griffinlim
from scipy.fftpack import idct
from scipy.linalg import pinv as pinv

filename = './audio.wav'
source_signal, sr = sf.read(filename)  # sr:sampling rate
print('Sampling rate={} Hz.'.format(sr))

# hyper parameters
frame_length = 512                    # Frame length(samples)
frame_step = 128                      # Step length(samples)
emphasis_coeff = 0.95                 # pre-emphasis parameter
num_bands = 64                        # Filter number = band number
num_bands_12 = 12
num_FFT = frame_length                # FFT freq-quantization
freq_min = 0
freq_max = int(0.5 * sr)
signal_length = len(source_signal)    # Signal length

# number of frames it takes to cover the entirety of the signal
num_frames = 1 + \
    int(np.ceil((1.0 * signal_length - frame_length) / frame_step))

##########################
'''
Part I:
(1) Perform STFT on the source signal to obtain one spectrogram (with the provided STFT() function)
(2) Pre-emphasize the source signal with pre_emphasis()
(3) Perform STFT on the pre-emphasized signal to obtain the second spectrogram
(4) Plot the two spectrograms together to observe the effect of pre-emphasis

hint for plotting:
you can use "plt.subplots()" to plot multiple figures in one.
you can use "axis.pcolor" of matplotlib in visualizing a spectrogram. 
'''
# YOUR CODE STARTS HERE:
spec = STFT(source_signal, num_frames, num_FFT, frame_step,
            frame_length, signal_length, verbose=True)

# pre-emphasized signal
signal_em = pre_emphasis(source_signal, coefficient=emphasis_coeff)

spec_amp = STFT(signal_em, num_frames, num_FFT, frame_step,
                frame_length, signal_length, verbose=False)         # spectrogram of the pre-emphasized signal

# plot the spectrogram of the original signal
# fig, (ax0, ax1) = plt.subplots(1, 2)

# ax0.pcolor(spec)                    # pcolor instead of plot
# ax0.set_title('Original signal')
# ax0.set_xlabel('frame')
# ax0.set_ylabel('frequency band')

# # plot the spectrogram of the pre-emphasized signal
# ax1.pcolor(spec_amp)
# ax1.set_title('Pre-emphasized signal')
# ax1.set_xlabel('frame')
# ax1.set_ylabel('frequency band')

# YOUR CODE ENDS HERE;
##########################

'''
Head to the import source 'Lab1_functions_student.py' to complete these functions:
mel2hz(), hz2mel(), get_filter_banks()
'''
# get Mel-scaled filter
fbanks = get_filter_banks(num_bands, num_FFT, sr,
                          freq_min, freq_max)       # 64 fbanks
fbanks_12 = get_filter_banks(
    num_bands_12, num_FFT, sr, freq_min, freq_max)  # 12 fbanks

x = np.arange(freq_min, freq_max, freq_max/(frame_step + 1))
##########################
'''
Part II:
(1) Convolve the pre-emphasized signal with the filter
(2) Convert magnitude to logarithmic scale
(3) Perform Discrete Cosine Transform (dct) as a process of information compression to obtain MFCC
    (already implemented for you, just notice this step is here and skip to the next step)
(4) Plot the filter banks alongside the MFCC
'''
# YOUR CODE STARTS HERE:
# convolution in time domain meaning multiplication in freq-domain
con = np.dot(fbanks, spec_amp)          # pre emphasized 64 fbanks
con_12 = np.dot(fbanks_12, spec_amp)    # pre emphasized 12 fbanks

con_nonpre_64 = np.dot(fbanks, spec)    # nonpre 64 fbanks
con_nonpre_12 = np.dot(fbanks_12, spec)  # nonpre 12 fbanks
# avoid getting 0 when perform log next

# 64 banks (pre)
con = np.where(con == 0, np.finfo(float).eps, con)
features = np.log(con)          # convert magnitude into log scale
features_T = features.T         # take transpose

# 12 banks (pre)
con_12 = np.where(con_12 == 0, np.finfo(float).eps, con_12)
features_12 = np.log(con_12)
features_12_T = features_12.T

# 64 banks (nonpre)
con_nonpre_64 = np.where(
    con_nonpre_64 == 0, np.finfo(float).eps, con_nonpre_64)
features_nonpre_64 = np.log(con_nonpre_64)
features_nonpre_64_T = features_nonpre_64.T

# 12 banks (nonpre)
con_nonpre_12 = np.where(
    con_nonpre_12 == 0, np.finfo(float).eps, con_nonpre_12)
features_nonpre_12 = np.log(con_nonpre_12)
features_nonpre_12_T = features_nonpre_12.T

# step(3): Discrete Cosine Transform

# pre 64 banks
MFCC = dct(features_T, norm='ortho')[
    :, :num_bands]  # perform DCT on the features

# pre 12 banks
MFCC_12 = dct(features_12_T, norm='ortho')[
    :, :num_bands_12]  # perform DCT on the features

# nonpre 64 banks
MFCC_nonpre_64 = dct(features_nonpre_64_T, norm='ortho')[
    :, :num_bands]  # perform DCT on the features

# nonpre 12 banks
MFCC_nonpre_12 = dct(features_nonpre_12_T, norm='ortho')[
    :, :num_bands_12]  # perform DCT on the features

# equivalent to Matlab dct(x)
# The numpy array [:,:] stands for everything from the beginning to end.

# plot demo (2)
fig, (ax0, ax1) = plt.subplots(1, 2)    # plot two subplots
ax0.pcolor(MFCC.T)
ax0.set_title('64 banks MFCC')
ax0.set_xlabel('frame')
ax0.set_ylabel('MFCC coefficient')

ax1.pcolor(MFCC_12.T)
ax1.set_title('12 banks MFCC')
ax1.set_xlabel('frame')
ax1.set_ylabel('MFCC coefficient')

# plot demo (1)
fig, (ax0, ax1) = plt.subplots(1, 2)

ax0.plot(MFCC[0])
ax0.set_title('64 banks MFCC of a random frame')
ax0.set_xlabel('Cepstral Coefficient')
ax0.set_ylabel('Magnitude')

ax1.plot(MFCC_12[0])
ax1.set_title('12 banks MFCC of a random frame')
ax1.set_xlabel('Cepstal Coefficient')
ax1.set_ylabel('Magnitude')

# YOUR CODE ENDS HERE;
##########################
############ADD THESE#################
'''
(1) Perform inverse DCT on MFCC (already done for you)
(2) Restore magnitude from logarithmic scale (i.e. use exponential)
(3) Invert the fbanks convolution
(4) Synthesize time-domain audio with Griffin-Lim
(5) Get STFT spectrogram of the reconstructed signal and compare it side by side with the original signal's STFT spectrogram
    (please convert magnitudes to logarithmic scale to better present the changes)
'''

# inverse DCT (done for you)
inv_DCT = idct(MFCC, norm='ortho')
print('64 banks Shape after iDCT:', inv_DCT.shape)

inv_DCT_12 = idct(MFCC_12, norm='ortho')
print('12 banks Shape after iDCT:', inv_DCT_12.shape)

inv_DCT_nonpre_64 = idct(MFCC_nonpre_64, norm='ortho')
print('64 banks nonpreemphasized Shape after iDCT:', inv_DCT_nonpre_64.shape)

inv_DCT_nonpre_12 = idct(MFCC_nonpre_12, norm='ortho')
print('12 banks non-preemphasized Shape after iDCT:', inv_DCT_nonpre_12.shape)

# mag scale restoration:
###################
# take exponential of inv_DCT
exp_MFCC = np.exp(inv_DCT)                     # 64 banks inverse
exp_MFCC_12 = np.exp(inv_DCT_12)               # 12 banks inverse

exp_MFCC_nonpre_64 = np.exp(inv_DCT_nonpre_64)  # nonpre 64 banks inverse
exp_MFCC_nonpre_12 = np.exp(inv_DCT_nonpre_12)  # nonpre 12 banks inverse
###################

# inverse convoluation against fbanks (mind the shapes of your matrices):
###################
# perform inverse convolution, pinv(): pseudo inverse matrix
# 64 banks inverse spectrogram (pre)
inv_spectrogram = pinv(fbanks) @ exp_MFCC.T
# 12 banks inverse spectrogram (pre)
inv_spectrogram_12 = pinv(fbanks_12) @ exp_MFCC_12.T

# 64 banks inverse spectrogram (nonpre)
inv_spec_nonpre_64 = pinv(fbanks) @ exp_MFCC_nonpre_64.T
# 12 banks inverse spectrogram (nonpre)
inv_spec_nonpre_12 = pinv(fbanks_12) @ exp_MFCC_nonpre_12.T
###################
# print shape of 64 banks and 12 banks
print('(64 banks) Shape after inverse convolution:', inv_spectrogram.shape)
print('(12 banks) Shape after inverse convolution:', inv_spectrogram_12.shape)

# signal restoration to time domain (You only have to finish griffinlim() in 'stft2audio_student.py'):

# demo (4)
inv_audio = griffinlim(inv_spectrogram, n_iter=32,
                       hop_length=frame_step, win_length=frame_length)      # perform griffinlim algorithm to reconstruct signal

inv_audio_12 = griffinlim(inv_spectrogram_12, n_iter=32,
                          hop_length=frame_step, win_length=frame_length)      # perform griffinlim algorithm to reconstruct signal

inv_audio_nonpre_64 = griffinlim(
    inv_spec_nonpre_64, n_iter=32, hop_length=frame_step, win_length=frame_length)

inv_audio_nonpre_12 = griffinlim(
    inv_spec_nonpre_12, n_iter=32, hop_length=frame_step, win_length=frame_length)

sf.write('reconstructed_64_pre.wav', inv_audio,
         samplerate=int(sr*512/frame_length))

sf.write('reconstructed_12_pre.wav', inv_audio_12,
         samplerate=int(sr*512/frame_length))

sf.write('reconstructed_64_nonpre.wav', inv_audio_nonpre_64,
         samplerate=int(sr*512/frame_length))

sf.write('reconstructed_12_nonpre.wav', inv_audio_nonpre_12,
         samplerate=int(sr*512/frame_length))

reconstructed_spectrum = STFT(inv_audio, num_frames, num_FFT,
                              frame_step, frame_length, len(inv_audio), verbose=False)

# scale and plot and compare original and reconstructed signals
# scale (done for you):
absolute_spectrum = np.where(                   # np.finfo(dtype): to get the 'dtype' data
    spec == 0, np.finfo(float).eps, spec)       # eps is a very small positive number
# it is used when a deenominator is possibly zero
absolute_spectrum = np.log(absolute_spectrum)
reconstructed_spectrum = np.where(
    reconstructed_spectrum == 0, np.finfo(float).eps, reconstructed_spectrum)

# log scale of the reconstructed signal
reconstructed_spectrum = np.log(reconstructed_spectrum)

# plot demo (3):
###################
fig, (ax0, ax1) = plt.subplots(1, 2)

ax0.pcolor(absolute_spectrum)
ax0.set_title('Original signal')
ax0.set_xlabel('frame')
ax0.set_ylabel('frequency band')

ax1.pcolor(reconstructed_spectrum)
ax1.set_title('Reconstructed signal')
ax1.set_xlabel('frame')
ax1.set_ylabel('frequency band')

plt.show()
###################
############ADD ABOVE#################
