# Various functinos related to audio signal processing

import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fftpack import fft, ifft
from numpy import hamming

# STFT, ISTFT ----------------------------------------------------
def sig2spec(src, fft_size, shift_size, window='hamming'):
    """
    Parameters
    ----------
    signal: input signal
    fft_size: frame length
    shift_size: frame shift
    window: window function

    Returns
    -------
    S: spectrogram of input signal (fft_size/2+1 x frame x ch)
    """

    # ファイルパス指定ならsf.read
    # 信号ならそのまま読込
    if type(src) is str:
        signal, _ = sf.read(src)
    else:
        signal = np.array(src)

    if window == 'hamming':
        window = hamming(fft_size+1)[:fft_size]
    else:
        print(window+' is not supported.')
        sys.exit(0)

    zeroPadSize = fft_size - shift_size
    length = signal.shape[0]
    frames = int(np.floor((length + fft_size - 1) / shift_size))
    I = int(fft_size/2 + 1)

    if len(signal.shape) == 1:
        # monoral
        signal = np.concatenate([np.zeros(zeroPadSize), signal, np.zeros(fft_size)])
        S = np.zeros([I, frames], dtype=np.complex128)

        for j in range(frames):
            sp = j * shift_size
            spectrum = fft(signal[sp: sp+fft_size] * window)
            S[:, j] = spectrum[:I]

        return S

    elif len(signal.shape) == 2:
        nch = signal.shape[1]
        signal = np.concatenate([np.zeros([zeroPadSize, nch]), signal, np.zeros([fft_size,nch])])
        S = np.zeros([I, frames, nch], dtype=np.complex128)

        for ch in range(nch):
            for j in range(frames):
                sp = j * shift_size
                spectrum = fft(signal[sp: sp+fft_size, ch] * window)
                S[:, j, ch] = spectrum[:I]

        return S

    else:
        raise ValueError('illegal signal dimension')


def show_spec(spec, fs=16000, fft_size=2048, shift_size=1024, xrange=None, yrange=None, crange=[-40,0], cbar=False, output=None, imshow=False, plot_type='log'):
    '''
    Parameters
    ----------------
    spec: Specetrogram (1ch absolute scale)
    fs: Sampling frequency
    fft_size: FFT length
    shift_size: overlap length
    xrange: Range of X axis (list [x min, x max])
    yrange: Range of Y axis (same format as xrange)
    crange: Range of Colorbar (list [min(db), max(db)])
    cbar: Whether show color bar or not
    output: Output file name 
    imshow: Preview in program (Does not work on remote env.)
    '''

    if len(spec.shape) == 1:
        spec = spec.reshape([spec.shape[0],1])
        I = spec.shape[0]
        J = 1
    else:
        I, J = spec.shape

    if fft_size is None:
        fft_size = (I - 1) * 2
    if shift_size is None:
        shift_size = fft_size // 2

    t = np.round((J * shift_size - fft_size + 1) / fs)+1

    # x-y axis range
    if xrange == None:
        x_min = 0
        x_max = t  
    else:
        x_min = xrange[0]
        x_max = xrange[1]

    if yrange == None:
        y_min = fs // 2
        y_max = 0
    else:
        y_min = yrange[1]
        y_max = yrange[0]

    ax_range = [x_min, x_max, y_min, y_max]

    plt.figure()

    if plot_type == 'log':
        epsilon = sys.float_info.epsilon
        S = np.where(spec < epsilon, spec+epsilon, spec) # フロアリング
        plt.imshow(20*np.log10(S), extent=ax_range, aspect='auto', vmin=crange[0], vmax=crange[1], cmap='cividis', interpolation='nearest')
    elif plot_type == 'default':
        plt.imshow(spec, extent=ax_range, aspect='auto', vmin=crange[0], vmax=crange[1], cmap='cividis', interpolation='nearest')

    if cbar:
        plt.colorbar()

    plt.gca().invert_yaxis()
    
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    if output != None:
        plt.savefig(output, bbox_inches='tight', dpi=300)
    
    if imshow is True:
        plt.show()

    plt.close()

def opt_syn_wnd(analysis_window, shift_size):
    fft_size = analysis_window.shape[0]
    synthesized_window = np.zeros(fft_size)
    for i in range(shift_size):
        amp = 0
        for j in range(1, int(fft_size / shift_size) + 1):
            amp += analysis_window[i + (j-1) * shift_size] ** 2
        for j in range(1, int(fft_size / shift_size) + 1):
            synthesized_window[i + (j-1) * shift_size] = analysis_window[i + (j-1) * shift_size] / amp

    return synthesized_window


def spec2sig(S, shift_size, window=None, len=None):
    """
    Parameters
    ------------------------
    S: STFT of input signal (fft_size/2+1 x frames x nch)
    shift_size: frame shift (default: fft_size/2)
    window: window function used in STFT (fft_size x 1) or choose used
            function from below.
            'hamming'    : Hamming window (default)
            'hann'       : von Hann window
            'rectangular': rectangular window
            'blackman'   : Blackman window
            'sine'       : sine window
    len: length of original signal (before STFT)
    
    Return
    ------------------------
    waveform: time-domain waveform of the input spectrogram (len x nch)
    """

    if window is None:
        fft_size = shift_size * 2
        window = hamming(fft_size+1)[:fft_size]

    if S.ndim == 2:
        freq, frames = S.shape
        fft_size = (freq-1) * 2
        inv_window = opt_syn_wnd(window, shift_size)
        spectrum = np.zeros(fft_size, dtype=np.complex128)

        tmp_signal = np.zeros([(frames - 1) * shift_size + fft_size])
        for j in range(frames):
            spectrum[:int(fft_size / 2) + 1] = S[:, j]
            spectrum[0] /= 2
            spectrum[int(fft_size / 2)] /= 2
            sp = j * shift_size
            tmp_signal[sp: sp + fft_size] += (np.real(ifft(spectrum, fft_size) * inv_window) * 2)

        waveform = tmp_signal[fft_size - shift_size: (frames - 1) * shift_size + fft_size]

    elif S.ndim == 3:
        freq, frames, nch = S.shape
        fft_size = (freq-1) * 2
        inv_window = opt_syn_wnd(window, shift_size)
        spectrum = np.zeros(fft_size, dtype=np.complex128)

        tmp_signal = np.zeros([(frames - 1) * shift_size + fft_size, nch])
        for ch in range(nch):
            for j in range(frames):
                spectrum[:int(fft_size / 2) + 1] = S[:, j, ch]
                spectrum[0] /= 2
                spectrum[int(fft_size / 2)] /= 2
                sp = j * shift_size
                tmp_signal[sp: sp + fft_size, ch] += (np.real(ifft(spectrum, fft_size) * inv_window) * 2)

        waveform = tmp_signal[fft_size - shift_size: (frames - 1) * shift_size + fft_size]

    if len:
        waveform = waveform[:len]

    return waveform