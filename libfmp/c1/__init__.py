from .c1s1_sheet_music import generate_sinusoid_pitches, \
    generate_shepard_tone, \
    generate_chirp_exp_octave, \
    generate_shepard_glissando

from .c1s2_symbolic_rep import csv_to_list, \
    midi_to_list, \
    xml_to_list, \
    list_to_csv, \
    visualize_piano_roll

from .c1s3_audio_rep import f_pitch, \
    difference_cents, \
    generate_sinusoid, \
    compute_power_db, \
    compute_equal_loudness_contour, \
    generate_chirp_exp, \
    generate_chirp_exp_equal_loudness, \
    compute_adsr, compute_envelope, \
    compute_plot_envelope, \
    generate_sinusoid_vibrato, \
    generate_sinusoid_tremolo, \
    generate_tone
