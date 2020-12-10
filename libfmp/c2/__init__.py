from .c2_complex import generate_figure, \
    plot_vector

from .c2_fourier import generate_matrix_dft, \
    generate_matrix_dft_inv, \
    dft, \
    idft, \
    fft, \
    ifft_noscale, \
    ifft, \
    stft_basic, \
    istft_basic, \
    stft, istft, \
    stft_convention_fmp

from .c2_interpolation import compute_f_coef_linear, \
    compute_f_coef_log, \
    interpolate_freq_stft

from .c2_interference import plot_interference, \
    generate_chirp_linear

from .c2_digitization import generate_function, \
    sampling_equidistant, \
    reconstruction_sinc, \
    plot_graph_quant_function, \
    quantize_uniform, \
    plot_signal_quant, \
    encoding_mu_law, \
    decoding_mu_law, \
    plot_mu_law, \
    quantize_nonuniform_mu
