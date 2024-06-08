from common.functions import sig2spec, show_spec
import soundfile as sf
import numpy as np

if __name__ == '__main__':

    wat_path = '../dataset/news/man/1/origin.wav'
    sig, sr = sf.read(wat_path)

    fft_size = 1024
    shift_size = 512

    S = sig2spec(sig, fft_size, shift_size, window='hamming')
    show_spec(np.abs(S), sr, fft_size, shift_size, imshow=True)