# -*- codingg: utf-8 -*-

import copy
import numpy as np
from scipy.fftpack import dct
from scipy.signal import stft
import matplotlib.pyplot as plt
import librosa


# cepstram coefficient like feature extraction class
class AcousticFeatures:
    def __init__(self, sound, fs):
        self.sound = sound
        self.fs = fs
        self.n_fft = 2048

    # mel-frequency cepstram coefficient 
    def extract_mfcc(self, window_second=0.02, stride_second=0.01, n_features=13):
        pre_emphasis_sound = self.__pre_emphasis(sound)
        win_length = int(window_second * self.fs)
        stride = int(stride_second * self.fs)

        ## create gammatone spectrogram
        f, t, spectrogram = stft(
            pre_emphasis_sound,
            nfft=self.n_fft,
            fs=self.fs,
            noverlap=stride,
            nperseg=win_length,
            window="hann",
        )

        filterbank, hz = self.mel_filterbank(n_channels=21)
        mel_spectrogram = np.dot(filterbank, spectrogram[1:])[:, :-1]
        log_power_melspectrogram = np.log10(np.abs(mel_spectrogram) ** 2)
        ## calculate cepstrum
        mfc = dct(log_power_melspectrogram)
        mfcc = mfc[:n_features]
        return mfcc, t

    # extract RASTA PLP
    def extract_rasta_plp(
        self,
        window_second=0.02,
        stride_second=0.01,
        n_features=13,
        do_rasta_filter=True,
    ):
        win_length = window_second * self.fs
        stride = stride_second * self.fs

        ## calculate bark power spectrogram
        f, t, spectrogram = stft(
            sound,
            nfft=self.n_fft,
            fs=self.fs,
            noverlap=stride,
            nperseg=win_length,
            window="hann",
        )
        filterbank, hz_center = self.bark_filterbank(n_channels=21)
        bark_spectrogram = np.dot(filterbank, spectrogram[1:])[:, :-1]
        log_power_spectrogram = np.log10(np.abs(bark_filterbank) ** 2)

        ## RASTA filtering
        if do_rasta_filter:
            rasta_filtered = self.__rasta_filtering(log_power_spectrogram)

        ## TODO: この後を実装
        pass

    # gammatone frequency cepstram coefficient
    def extract_gfcc(self, window_second=0.02, stride_second=0.01, n_features=13):
        pre_emphasis_sound = self.__pre_emphasis(sound)
        win_length = int(window_second * self.fs)
        stride = stride_second * self.fs

        ## create gammatone spectrogram
        f, t, spectrogram = stft(
            pre_emphasis_sound,
            nfft=self.n_fft,
            fs=self.fs,
            noverlap=stride,
            nperseg=win_length,
            window="hann",
        )
        filterbank, hz_center = self.gammachirp_filterbank(n_channels=21)
        gammatone_spectrogram = np.dot(filterbank, spectrogram[1:])[:, :21]
        log_power_gtspectrogram = np.log10(np.abs(gammatone_spectrogram) ** 2)

        ## calculate cepstrum
        gfc = dct(log_power_gtspectrogram)
        gfcc = gfc[:n_features]
        return gfcc, t

    # gammatone frequency modulation coefficient 
    def extract_gfmc(self, window_second=0.02, stride_second=0.01, n_features=13):
        pre_emphasis_sound = self.__pre_emphasis(sound)
        win_length = window_second * self.fs
        stride = stride_second * self.fs

        ## create gammatone spectrogram
        f, t, spectrogram = stft(
            pre_emphasis_sound,
            nfft=self.n_fft,
            fs=self.fs,
            noverlap=stride,
            nperseg=win_length,
            window="hann",
        )
        filterbank, hz_center = self.gammachirp_filterbank(n_channels=31)
        gammatone_spectrogram = np.dot(filterbank, spectrogram[1:])[:, :-1]
        log_power_gtspectrogram = np.log10(np.abs(gammatone_spectrogram) ** 2)

        ## calculate cepstrum
        gfc = dct(log_power_gtspectrogram)
        gfmc = []
        for gfc_row in gfc:
            f1, f2, gfm = stft(gfc_row, nperseg=100, noverlap=90)
            gfm_log_power = np.log10(np.abs(np.array(gfm[:, 0])) ** 2)
            gfmc.append(gfm_log_power)

        return np.array(gfmc)

    # zero-crossing and peak-amplitude
    def extract_zcpa(self):
        pass

    # power normalized cepstram coefficient
    def extract_pncc(self, n_features=13, alpha=0.9, beta=0.5, lamda=0.85, mu=0.2):
        pre_emphasis_sound = self.__pre_emphasis(self.sound)
        win_length = 0.0256 * self.fs
        stride = 0.01 * self.fs

        ## create gammatone spectrogram
        f, t, spectrogram = stft(
            pre_emphasis_sound,
            nfft=self.n_fft,
            fs=self.fs,
            noverlap=stride,
            nperseg=win_length,
            window="hamming",
        )
        
        filterbank, hz_center = self.gammachirp_filterbank(n_channels=40)
        gammatone_spectrogram = np.dot(filterbank, spectrogram[1:])[:, :-1]
        log_power_gtspectrogram = np.log10(np.abs(gammatone_spectrogram) ** 2)

        ## calculate medium time power
        num = 5
        window = np.ones(num).astype(np.float32) / num
        mid_power = np.empty_like(log_power_gtspectrogram)
        for idx in range(log_power_gtspectrogram.shape[0]):
            mid_power[idx] = np.convolve(window, log_power_gtspectrogram[idx], mode="same")
        
        filtered_mid_power = copy.deepcopy(mid_power)
        
        ## asymmetric noise filtering and temporal masking
        for row in range(1, filtered_mid_power.shape[0]):
            for col in range(filtered_mid_power.shape[1]):
                if filtered_mid_power[row, col] >= filtered_mid_power[row-1, col]:
                    filtered_mid_power[row, col] = alpha * filtered_mid_power[row-1, col] 
                    + (1-alpha) * filtered_mid_power[row, col]
                else:
                    filtered_mid_power[row, col] = beta * filtered_mid_power[row-1, col] 
                    + (1-beta) * filtered_mid_power[row, col]
                
                if mid_power[row, col] >= lamda * filtered_mid_power[row, col]:
                    filtered_mid_power[row, col] = mid_power[row, col]
                else:
                    filtered_mid_power[row, col] = mu * filtered_mid_power[row, col]
                
        ## spectral smoothing
        num = 9
        window = np.ones(num).astype(np.float32) / num
        smoothed_power = np.empty_like(mid_power)
        for idx in range(mid_power.shape[1]):
            mid_power[:, idx] = np.convolve(window, mid_power[:, idx] / filtered_mid_power[:, idx], mode="same")
        normalized_power = log_power_gtspectrogram * mid_power

        ## mean power normalization
        lambda_mean = 0.999
        mean_power = np.empty(normalized_power.shape[0])
        for idx in range(normalized_power.shape[0]):
            if idx == 0:
                mean_power[idx] = np.mean(normalized_power[idx])
            else:
                mean_power[idx] = lambda_mean * mean_power[idx-1] + (1-lambda_mean) * np.mean(normalized_power[idx])
        normalized_power = normalized_power / mean_power[:, np.newaxis]
        # normalized_power = np.power(normalized_power, -15)

        pnc = dct(normalized_power)
        pncc = pnc[:n_features]
        return pncc, t

    # amplitude modulation spectrogram
    def extract_ams(self):
        pass

    # ssf
    def extract_ssf(self):
        pass
    
    # multi-resolution cochleagram 
    def extract_mrcg(self):
        pass

    # calculate mel filterbank
    def mel_filterbank(self, n_channels=20):
        nyq = self.fs / 2
        melmax = self.__hz2mel(nyq)

        n_max = self.n_fft / 2
        delta_f = fs / self.n_fft
        delta_mel = melmax / (n_channels + 1)
        mel_centers = np.arange(1, n_channels + 1) * delta_mel

        hz_centers = self.__mel2hz(mel_centers)
        index_center = np.round(hz_centers / delta_f)
        index_start = np.hstack(([0], index_center[0 : n_channels - 1]))
        index_stop = np.hstack((index_center[1:n_channels], [n_max]))

        filterbank = np.zeros((n_channels, int(n_max)))
        for c in np.arange(0, n_channels):
            # 三角フィルタの左の直線の傾きから点を求める
            increment = 1.0 / (index_center[c] - index_start[c])
            for i in np.arange(index_start[c], index_center[c]):
                filterbank[c, int(i)] = (i - index_start[c]) * increment
            # 三角フィルタの右の直線の傾きから点を求める
            decrement = 1.0 / (index_stop[c] - index_center[c])
            for i in np.arange(index_center[c], index_stop[c]):
                filterbank[c, int(i)] = 1.0 - ((i - index_center[c]) * decrement)

        return filterbank, hz_centers

    # calculate bark filterbank
    def bark_filterbank(self, n_channels=20):
        nyq = self.fs / 2
        bark_max = self.__hz2bark(nyq)

        n_max = self.n_fft / 2
        delta_f = fs / self.n_fft
        delta_bark = bark_max / (n_channels + 1)
        bark_centers = np.arange(1, n_channels + 1) * delta_bark

        hz_centers = self.__bark2hz(bark_centers)
        index_center = np.round(hz_centers / delta_f)
        index_start = np.hstack(([0], index_center[0 : n_channels - 1]))
        index_stop = np.hstack((index_center[1:n_channels], [n_max]))

        filterbank = np.zeros((n_channels, int(n_max)))
        for c in np.arange(0, n_channels):
            # 三角フィルタの左の直線の傾きから点を求める
            increment = 1.0 / (index_center[c] - index_start[c])
            for i in np.arange(index_start[c], index_center[c]):
                filterbank[c, int(i)] = (i - index_start[c]) * increment
            # 三角フィルタの右の直線の傾きから点を求める
            decrement = 1.0 / (index_stop[c] - index_center[c])
            for i in np.arange(index_center[c], index_stop[c]):
                filterbank[c, int(i)] = 1.0 - ((i - index_center[c]) * decrement)

        return filterbank, hz_centers

    def gammachirp_filterbank(self, n_channels=20):
        nyq = self.fs / 2
        gamma_max = self.__hz2gamma(nyq)

        n_max = self.n_fft / 2
        delta_f = fs / self.n_fft
        delta_gamma = gamma_max / (n_channels + 1)
        gamma_centers = np.arange(1, n_channels + 1) * delta_gamma

        hz_centers = self.__gamma2hz(gamma_centers)
        index_center = np.round(hz_centers / delta_f)
        index_start = np.hstack(([0], index_center[0 : n_channels - 1]))
        index_stop = np.hstack((index_center[1:n_channels], [n_max]))

        filterbank = np.zeros((n_channels, int(n_max)))
        for c in np.arange(0, n_channels):
            # 三角フィルタの左の直線の傾きから点を求める
            increment = 1.0 / (index_center[c] - index_start[c])
            for i in np.arange(index_start[c], index_center[c]):
                filterbank[c, int(i)] = (i - index_start[c]) * increment
            # 三角フィルタの右の直線の傾きから点を求める
            decrement = 1.0 / (index_stop[c] - index_center[c])
            for i in np.arange(index_center[c], index_stop[c]):
                filterbank[c, int(i)] = 1.0 - ((i - index_center[c]) * decrement)

        return filterbank, hz_centers

    def __hz2mel(self, freq):
        return 1127.01048 * np.log(freq / 700.0 + 1.0)

    def __mel2hz(self, freq):
        return 700.0 * np.exp(freq / 1127.01048)

    def __hz2bark(self, freq):
        return 6.0 * np.arcsinh(freq / 600.0)

    def __bark2hz(self, freq):
        return 600.0 * np.sinh(freq / 6.0)

    def __hz2gamma(self, freq):
        return 24.7 * (4.37 * freq / 1000.0 + 1.0)

    def __gamma2hz(self, freq):
        return (1000.0 / 4.37) * (freq / 24.7 - 1.0)

    def __rasta_filtering(self, data):
        inverse = 1.0 / data
        return (
            0.1
            * (2 + inverse - inverse ** 3 - 2 * inverse ** 4)
            / (inverse ** 4 * (1 - 0.98 * inverse))
        )

    def __pre_emphasis(self, z):
        return 1 - 0.97 * z


if __name__ == "__main__":
    sound, fs = librosa.load("../test_sound.wav", sr=44100)
    extractor = AcousticFeatures(sound, fs)
    pncc, t = extractor.extract_pncc()
    plt.imshow(pncc)
    plt.show()
