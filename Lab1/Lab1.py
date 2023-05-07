import numpy as np
import matplotlib.pyplot as plt

import signals

class Lab1:
    def __init__(self):
        self.__inputSignal = np.array(signals.Input_1kHz_15kHz)
        self.__impulseResponse = np.array(signals.Impulse_response)
        self.__ECG = np.array(signals.ECG)
                
        self.__inputSignalFreqResponse = np.fft.fft(self.__inputSignal)
        
        self.__impulseFreqresponse = np.fft.fft(self.__impulseResponse)
        
    def plotInputSignals(self):
        plt.figure(figsize = (16,3))
        plt.stem(self.__inputSignal, markerfmt= " ")
        plt.title("Input signal")
        plt.show()
        
    def convertInputSignalToFrequency(self):
        plt.figure(figsize = (16,9))
        
        plt.subplot(4, 1, 1)
        plt.stem(self.__inputSignalFreqResponse.real, "r", label= "Real", markerfmt= " ")
        plt.legend()
        
        plt.title("Input Signal Frequency")
        
        plt.subplot(4, 1, 2)
        plt.stem(self.__inputSignalFreqResponse.imag, "g", label= "Imaginary", markerfmt= " ")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.stem(np.abs(self.__inputSignalFreqResponse), "b", label= "Magnitude", markerfmt= " ")
        plt.legend()
        
        plt.subplot(4, 1, 4)
        plt.stem(np.angle(self.__inputSignalFreqResponse), "black", label= "Phase", markerfmt= " ")
        plt.legend()
        
        plt.show()
    
    def usingIFFTinputSignal(self):
        self.__signal_w_ifft = np.fft.ifft(self.__inputSignalFreqResponse)
        
        plt.figure(figsize = (16,3))
        
        plt.stem(self.__signal_w_ifft, markerfmt= " ")
        plt.title("Inverese FFT")
        
        plt.show()
    
    def plotImpulseResponse(self):
        plt.figure(figsize = (16,3))
        plt.stem(self.__impulseResponse, markerfmt= " ")
        plt.title("Impulse Response")
        plt.show()
    
    def convertImpulseToFrequency(self):
        plt.figure(figsize = (16,9))
        
        plt.subplot(4, 1, 1)
        plt.stem(self.__impulseFreqresponse.real, "r", label= "Real", markerfmt= " ")
        plt.legend()
        
        plt.title("Impulse Response Frequency")
        
        plt.subplot(4, 1, 2)
        plt.stem(self.__impulseFreqresponse.imag, "g", label= "Imaginary", markerfmt= " ")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.stem(np.abs(self.__impulseFreqresponse), "b", label= "Magnitude", markerfmt= " ")
        plt.legend()
        
        plt.subplot(4, 1, 4)
        plt.stem(np.angle(self.__impulseFreqresponse), "black", label= "Phase", markerfmt= " ")
        plt.legend()
        
        plt.show()
    
    def calcConvolutionSum(self):
        self.__convolutionSum = np.convolve(self.__inputSignal, self.__impulseResponse)
        
        plt.figure(figsize = (16,3))
        
        plt.stem(self.__convolutionSum, markerfmt= " ")
        plt.title("Convolution Sum")
        
        plt.show()
    
    def convertConvolToFrequency(self):
        self.__convolFreqresponse = np.fft.fft(self.__convolutionSum)
        
        plt.figure(figsize = (16,9))
        
        plt.subplot(4, 1, 1)
        plt.stem(self.__convolFreqresponse.real, "r", label= "Real", markerfmt= " ")
        plt.legend()
        
        plt.title("Convolution Sum Frequency")
        
        plt.subplot(4, 1, 2)
        plt.stem(self.__convolFreqresponse.imag, "g", label= "Imaginary", markerfmt= " ")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.stem(np.abs(self.__convolFreqresponse), "b", label= "Magnitude", markerfmt= " ")
        plt.legend()
        
        plt.subplot(4, 1, 4)
        plt.stem(np.angle(self.__convolFreqresponse), "black", label= "Phase", markerfmt= " ")
        plt.legend()
        
        plt.show()
    
    def calcFreqMultiplication(self):
        inputSignal_for_mult = np.fft.fft(self.__inputSignal, len(self.__convolutionSum))
        impulse_for_mult = np.fft.fft(self.__impulseResponse, len(self.__convolutionSum))
        
        Freq_domain_multiplication = inputSignal_for_mult * impulse_for_mult
        
        # Inverse FFT to obtain the resulting time-domain signal
        self.__Freq_mult_time_domain_signal = np.fft.ifft(Freq_domain_multiplication)
        
        plt.figure(figsize = (16,3))
        
        plt.stem(self.__Freq_mult_time_domain_signal, markerfmt= " ")
        plt.title("Frequency multiplication")
        
        plt.show()
        
    def convertFreqMultToFrequency(self):
        self.__freqMultFreqresponse = np.fft.fft(self.__Freq_mult_time_domain_signal)
        
        plt.figure(figsize = (16,9))
        
        plt.subplot(4, 1, 1)
        plt.stem(self.__freqMultFreqresponse.real, "r", label= "Real", markerfmt= " ")
        plt.legend()
        
        plt.title("Frequency Multiplication Freq")
        
        plt.subplot(4, 1, 2)
        plt.stem(self.__freqMultFreqresponse.imag, "g", label= "Imaginary", markerfmt= " ")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.stem(np.abs(self.__freqMultFreqresponse), "b", label= "Magnitude", markerfmt= " ")
        plt.legend()
        
        plt.subplot(4, 1, 4)
        plt.stem(np.angle(self.__freqMultFreqresponse), "black", label= "Phase", markerfmt= " ")
        plt.legend()
        
        plt.show()
    
    
        
        
run = Lab1()

# 2.Signals
run.plotInputSignals()
run.convertInputSignalToFrequency()
run.usingIFFTinputSignal()


# 3.Systems
run.plotImpulseResponse()
run.convertImpulseToFrequency()

run.calcConvolutionSum()
run.convertConvolToFrequency()

run.calcFreqMultiplication()
run.convertFreqMultToFrequency()


# 4.Filtering
