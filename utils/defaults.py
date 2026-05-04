# utils/defaults.py

DEFAULT_NEB = {
    # ------------------------------------------------------------------
    # NEB geometry / optimizer
    # ------------------------------------------------------------------
    "target_neb_beads": 15,
    "neb_spring_constant": 0.1,
    "neb_method": "improvedtangent",
    "neb_optimizer": "FIRE",
    "neb_fmax": 0.05,
    "neb_steps": 1000,

    # If True, standard NEB run folder is simply:
    #   neb_standard/
    # If False:
    #   neb_standard_<hash>/
    "standard_neb_native_name": True,

    # ------------------------------------------------------------------
    # Runtime muting flags
    # ------------------------------------------------------------------
    # Skip only standard NEB.
    # "standart" spelling is kept for compatibility with old configs.
    "mute_standard_neb": True,
    "mute_standart_neb": False,

    # Skip all actual NEB calculations.
    # Guesses are still prepared.
    "mute_all_neb": False,

    # Alias-style flag: same practical meaning as mute_all_neb=True.
    "prepare_guesses_only": False,

    # Dynamic custom-guess budgeting after standard NEB reconnaissance.
    "GALIMA_NUSIRASYNETI": False,
}


DEFAULT_BRUTE = {
    # ------------------------------------------------------------------
    # Wave 1 brute-force / relaxation defaults
    # ------------------------------------------------------------------
    "image_pool_selection": "rmsd",
    "n_interpolated": 7,
    "exclude_ends": 0,

    "brute_force_laps": 3,
    "k_select_laps": [8, 7, 7],
    "zoom_radius_laps": [2, 1],
    "brute_top_n": 300,

    # ------------------------------------------------------------------
    # Wave 2 overrides
    # These override base keys only for method="wave2".
    # ------------------------------------------------------------------
    "n_interpolated_w2": 8,
    "brute_force_laps_w2": 3,
    "k_select_laps_w2": [7, 6, 6],
    "zoom_radius_laps_w2": [2, 1],
    "brute_top_n_w2": 300,

    # ------------------------------------------------------------------
    # Brute-force scoring
    # ------------------------------------------------------------------
    # Old misspelled key is kept for compatibility.
    "even_coeficient": 0.5,
    "even_coefficient": 0.5,
    "even_window": 1,

    # Which guess methods to prepare/run.
    # Possible: ["wave1"], ["wave2"], ["wave1", "wave2"].
    # If wave2 is requested, wave1 is prepared internally as source.
    "custom_guess_methods": ["wave1"],

    # ------------------------------------------------------------------
    # Degenerate-combo rescue
    # ------------------------------------------------------------------
    "degenerate_extra_zoom": 2,
    "degenerate_zero_fraction": 0.75,
    "degenerate_max_unique": 2,

    # ------------------------------------------------------------------
    # Retry / quality legacy knobs
    # ------------------------------------------------------------------
    "quality_max_step_cv": 1.0,
    "retry_even_window": 2,
    "retry_even_coeficient": 0.5,
    "retry_even_coefficient": 0.5,
    "retry_degenerate_extra_zoom": 4,

    # ------------------------------------------------------------------
    # Champion Arena switch
    # ------------------------------------------------------------------
    # False:
    #   wave1concoursed / wave2concoursed
    #   brute archive -> champion arena -> best candidate
    #
    # True:
    #   wave1basic / wave2basic
    #   brute best -> direct reparametrization to target_neb_beads
    "mute_concours": False,

    # ------------------------------------------------------------------
    # Champion Arena candidate selection
    # ------------------------------------------------------------------
    "reparam_target_candidates": 15,
    "reparam_diversity_min_diff": 0.35,
    "reparam_stage1_keep": 5,

    "reparam_energy_wildcards": 0,
    "reparam_energy_wildcard_pool": 300,
    "reparam_energy_wildcard_min_diff": 0.25,

    "reparam_scf_check": True,
    "reparam_recompute_originals_for_final": True,
    "reparam_spike_threshold_kcal": 10.0,

    # ------------------------------------------------------------------
    # Champion Arena soft physical score weights
    # ------------------------------------------------------------------
    "reparam_w_ea": 0.5,
    "reparam_w_jump": 1.0,
    "reparam_w_perp_stage1": 10.0,
    "reparam_w_perp_stage2": 25.0,
    "reparam_w_fmax_stage1": 2.0,
    "reparam_w_fmax_stage2": 4.0,

    # ------------------------------------------------------------------
    # Champion Arena soft limits
    # ------------------------------------------------------------------
    "reparam_max_allowed_ea_kcal": 80.0,
    "reparam_max_allowed_jump_kcal": 25.0,
    "reparam_max_allowed_step_cv": 0.7,
    "reparam_max_allowed_step_ratio": 3.0,
    "reparam_max_allowed_wiggle_ratio": 2.0,

    # ------------------------------------------------------------------
    # Missing-gradient fallbacks / penalties
    # ------------------------------------------------------------------
    "reparam_missing_perp_default_evA": 50.0,
    "reparam_missing_fmax_default_evA": 80.0,
    "reparam_missing_perp_penalty": 50.0,
    "reparam_missing_fmax_penalty": 25.0,

    # ------------------------------------------------------------------
    # Legacy Champion Arena keys
    # Kept so old configs / hashes do not break.
    # New score does not really rely on these.
    # ------------------------------------------------------------------
    "reparam_assume_single_barrier": True,
    "reparam_max_major_peaks": 1,
    "reparam_max_post_peak_uphill_kcal": 3.0,
    "reparam_max_secondary_rebound_kcal": 3.0,
    "reparam_max_pre_peak_downhill_kcal": 3.0,
    "reparam_max_allowed_fmax_evA": 30.0,

    # ------------------------------------------------------------------
    # Cache trust
    # ------------------------------------------------------------------
    "trust_existing_raw_npz": False,
    "trust_existing_clean_npz": False,

    # ------------------------------------------------------------------
    # Rebuild flags
    # Not supposed to be part of stable cache hash.
    # ------------------------------------------------------------------
    "force_rebuild_custom_guess": False,
    "force_rebuild_wave1": False,
    "force_rebuild_wave2": False,
}


DEFAULT_RELAX = {
    # ------------------------------------------------------------------
    # Relaxation budgets
    # ------------------------------------------------------------------
    "relax_max_scf_calls": 100,
    "relax_max_scf_calls_w2": 100,

    # ------------------------------------------------------------------
    # Clean-pool filtering
    # ------------------------------------------------------------------
    "clean_min_keep_target": 30,
    "clean_rescue_min_keep": 10,
    "clean_rescue_max_jump": 0.01,
    "max_force_threshold": 50.0,

    # ------------------------------------------------------------------
    # Endpoint relaxation
    # ------------------------------------------------------------------
    "endpoint_relax_cycles": 2,
    "endpoint_max_scf_calls": None,
    "endpoint_relax_stop_kcal": 0.005,
}