from .b_audio import read_audio, \
    write_audio, \
    audio_player_list

from .b_plot import plot_signal, \
    plot_matrix, \
    plot_chromagram, \
    compressed_gray_cmap, \
    MultiplePlotsWithColorbar, \
    plot_annotation_line, \
    plot_annotation_line_overlay, \
    plot_annotation_multiline, \
    plot_segments, \
    plot_segments_overlay, \
    color_argument_to_dict

from .b_layout import FloatingBox

from .b_annotation import read_csv, \
    write_csv, \
    cut_audio, \
    cut_csv_file

from .b_sonification import list_to_chromagram, \
    generate_shepard_tone, \
    sonify_chromagram, \
    sonify_chromagram_with_signal, \
    list_to_pitch_activations, \
    sonify_pitch_activations, \
    sonify_pitch_activations_with_signal


#from .b_sonifications import save_to_csv, load_from_csv, sonification_librosa, sonification_own, sonification_hpss_lab
# Requires files "data/B/plato.wav" and so on, which are not part of libfmp -> generates error
