from typing import Dict, Any

from utils.helpers import get_fully_loaded_ratio


def build_assumptions(
    startup_profile: Dict[str, Any] | None = None,
    scenario_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:

    startup_profile = startup_profile or {}

    fully_loaded_ratio = get_fully_loaded_ratio(startup_profile)

    return {
        "fully_loaded_ratio": fully_loaded_ratio,
        "burn_window_months": 3,
        "one_time_percentile": 0.90,
        "scenario_overrides": scenario_overrides or {},
    }