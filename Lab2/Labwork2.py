import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import firwin, lfilter, butter, filtfilt
from scipy.io import wavfile

import signals

class lab2:
    def __init__(self):
        self.__InputSignal = signals.Input_1kHz_15kHz
        self.__ImpulseResponse = signals.Impulse_response
        self.__ECG = signals.ECG
    
    def FIR(self):
        plt.figure(figsize= (14, 10))

        plt.subplot(2, 1, 1)
        hamming_filter = firwin(13, 0.3, window = "hamming")
        self.__FIRed1 = lfilter(hamming_filter, 1.0, self.__InputSignal)
        plt.plot(self.__InputSignal)
        plt.plot(self.__FIRed1)
        plt.title("Hamming Lowpass")

        plt.subplot(2, 1, 2)
        hamming_filter = firwin(13, 0.3, window = "hamming", pass_zero='highpass')
        self.__FIRed2 = lfilter(hamming_filter, 1.0, self.__InputSignal)
        plt.plot(self.__InputSignal)
        plt.plot(self.__FIRed2)
        plt.title("Hamming Highpass")

        plt.show()

    def FIR_freq_representation(self):
        plt.figure(figsize= (16, 5))
        y1 = np.fft.fftshift(abs(np.fft.fft(self.__InputSignal)))

        plt.subplot(1, 2, 1)
        
        y2 = np.fft.fftshift(abs(np.fft.fft(self.__FIRed1)))
        plt.title("Hamming Lowpass")
        plt.plot(y1)
        plt.plot(y2)
        
        plt.subplot(1, 2, 2)
        y3 = np.fft.fftshift(abs(np.fft.fft(self.__FIRed2)))
        plt.title("Hamming Highpass")

        plt.plot(y1)
        plt.plot(y3)
        plt.show()

    def IIR(self):
        # Filter requirements.
        T = 5.0         # Sample Period
        fs = 30.0       # sample rate, Hz
        cutoff = 2      # desired cutoff frequency of the filter, Hz , slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 2       # sin wave can be approx represented as quadratic
        n = int(T * fs) # total number of samples

        def butter_lowpass_filter(data, cutoff, order):
            normal_cutoff = cutoff / nyq
            # Get the filter coefficients 
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data)
            return y

        def butter_highpass_filter(data, cutoff, order):
            normal_cutoff = cutoff / nyq
            # Get the filter coefficients 
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            y = filtfilt(b, a, data)
            return y

        self.__IIRed1 = butter_lowpass_filter(self.__InputSignal, cutoff, order)
        self.__IIRed2 = butter_highpass_filter(self.__InputSignal, cutoff, order)

        plt.figure(figsize= (14, 6))

        plt.subplot(2, 1, 1)
        plt.plot(self.__InputSignal)
        plt.plot(self.__IIRed1)
        plt.title("Butter Lowpass")

        plt.subplot(2, 1, 2)
        plt.plot(self.__InputSignal)
        plt.plot(self.__IIRed2)
        plt.title("Butter Highpass")

        plt.show()
    
    def IIR_freq_representation(self):
        plt.figure(figsize= (16, 5))
        y1 = np.fft.fftshift(abs(np.fft.fft(self.__InputSignal)))

        plt.subplot(1, 2, 1)
        
        y2 = np.fft.fftshift(abs(np.fft.fft(self.__IIRed1)))
        plt.title("Butter Lowpass")
        plt.plot(y1)
        plt.plot(y2)
        
        plt.subplot(1, 2, 2)
        y3 = np.fft.fftshift(abs(np.fft.fft(self.__IIRed2)))
        plt.title("Butter Highpass")

        plt.plot(y1)
        plt.plot(y3)
        plt.show()

    def noiseRemoval(self):
        samplerate, data = wavfile.read("Lab2/SoundRecord.wav")
        data = data.reshape(-1)

        # Filter requirements.
        T = 5.0         # Sample Period
        fs = 30.0       # sample rate, Hz
        cutoff = 1.3     # desired cutoff frequency of the filter, Hz , slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 2       # sin wave can be approx represented as quadratic
        n = int(T * fs) # total number of samples

        def butter_lowpass_filter(data, cutoff, order):
            normal_cutoff = cutoff / nyq
            # Get the filter coefficients 
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data)
            return y
        
        self.__noiseReduced = butter_lowpass_filter(data, cutoff, order)

        plt.subplot(2, 1, 1)
        plt.plot(data)
        plt.title("Original Sound")

        plt.subplot(2, 1, 2)
        plt.plot(self.__noiseReduced)
        plt.title("Noise Removed")

        plt.show()
        
        self.__noiseReduced = np.int16(self.__noiseReduced)
        wavfile.write("Lab2/SoundNoiseRemoval.wav", samplerate, self.__noiseReduced)

run = lab2()
# Part 1 + 2
run.FIR()
run.FIR_freq_representation()
run.IIR()
run.IIR_freq_representation()

# Part 3
run.noiseRemoval()