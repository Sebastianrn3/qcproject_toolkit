DEFAULT_NEB = {
    "target_neb_beads": 15,
    "neb_spring_constant": 0.1,
    "neb_method": "improvedtangent",
    "neb_optimizer": "FIRE",
    "neb_fmax": 0.05,
    "neb_steps": 1000,

    "standard_neb_native_name": True,
    "mute_standard_neb": True,
    "mute_all_neb": False, #only create guesses
    "prepare_guesses_only": False,

}


DEFAULT_BRUTE = {
    "image_pool_selection": "rmsd",
    "n_interpolated": 7,
    "exclude_ends": 1,

    "brute_force_laps": 3,
    "k_select_laps": [8, 8, 7],
    "zoom_radius_laps": [2, 1],
    "brute_top_n": 300,


    "even_coefficient": 0.2,
    "even_window": 1,

    "custom_guess_methods": ["wave1", "wave2"],#"wave1"

    "degenerate_extra_zoom": 2,
    "degenerate_zero_fraction": 0.75,
    "degenerate_max_unique": 2,

    "trust_existing_raw_npz": True,
    "trust_existing_clean_npz": True,

    "force_rebuild_custom_guess": False,
    "force_rebuild_wave1": False,
    "force_rebuild_wave2": False,
}


DEFAULT_RELAX = {
    "relax_max_scf_calls": 1000,
    "relax_max_scf_calls_w2": 1000,

    "clean_rescue_min_keep": 10,
    "clean_rescue_max_jump": 0.01,
    "max_force_threshold": 50.0,

    "endpoint_relax_cycles": 2,
    "endpoint_max_scf_calls": None,
    "endpoint_relax_stop_kcal": 0.005,
}
