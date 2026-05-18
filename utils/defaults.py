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

    # Dynamic custom-guess budgeting after standard NEB reconnaissance.
    "GALIMA_NUSIRASYNETI": False,
}


DEFAULT_BRUTE = {
    # ------------------------------------------------------------------
    # Wave 1 brute-force / relaxation defaults
    # ------------------------------------------------------------------
    "image_pool_selection": "rmsd",
    "n_interpolated": 7,
    "exclude_ends": 1,

    "brute_force_laps": 3,
    "k_select_laps": [8, 8, 7],
    "zoom_radius_laps": [2, 1],
    "brute_top_n": 300,


    "even_coefficient": 0.2,
    "even_window": 9,

    "custom_guess_methods": ["wave1","wave2"],#"wave1"

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
    "retry_even_window": 1,
    "retry_even_coefficient": 0.5,
    "retry_degenerate_extra_zoom": 4,


    "mute_concours": True,

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

    "reparam_w_ea": 0.5,
    "reparam_w_jump": 1.0,
    "reparam_w_perp_stage1": 10.0,
    "reparam_w_perp_stage2": 25.0,
    "reparam_w_fmax_stage1": 2.0,
    "reparam_w_fmax_stage2": 4.0,

    "reparam_max_allowed_ea_kcal": 80.0,
    "reparam_max_allowed_jump_kcal": 25.0,
    "reparam_max_allowed_step_cv": 0.7,
    "reparam_max_allowed_step_ratio": 3.0,
    "reparam_max_allowed_wiggle_ratio": 2.0,

    "reparam_missing_perp_default_evA": 50.0,
    "reparam_missing_fmax_default_evA": 80.0,
    "reparam_missing_perp_penalty": 50.0,
    "reparam_missing_fmax_penalty": 25.0,

    "reparam_assume_single_barrier": True,
    "reparam_max_major_peaks": 1,
    "reparam_max_post_peak_uphill_kcal": 3.0,
    "reparam_max_secondary_rebound_kcal": 3.0,
    "reparam_max_pre_peak_downhill_kcal": 3.0,
    "reparam_max_allowed_fmax_evA": 30.0,

    # ------------------------------------------------------------------
    # Cache trust
    # ------------------------------------------------------------------
    "trust_existing_raw_npz": True,
    "trust_existing_clean_npz": True,

    # ------------------------------------------------------------------
    "force_rebuild_custom_guess": False,
    "force_rebuild_wave1": False,
    "force_rebuild_wave2": False,
}


DEFAULT_RELAX = {
    # ------------------------------------------------------------------
    # Relaxation budgets
    # ------------------------------------------------------------------
    "relax_max_scf_calls": 1000,
    "relax_max_scf_calls_w2": 1000,

    # ------------------------------------------------------------------
    # Clean-pool filtering
    # ------------------------------------------------------------------
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