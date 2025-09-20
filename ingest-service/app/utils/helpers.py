# helpers.py

def effective_cv(coal_cv: float, af_cv: float, af_pct: float) -> float:
    s = max(0.0, min(100.0, af_pct)) / 100.0
    return (1 - s) * coal_cv + s * af_cv

def heat_ratio(coal_cv: float, af_cv: float, af_pct: float) -> float:
    eff = effective_cv(coal_cv, af_cv, af_pct)
    return eff / coal_cv

def predicted_temp_with_af(base_temp: float, coal_cv: float, af_cv: float, af_pct: float, deg_per_pct: float = 2.0):
    r = heat_ratio(coal_cv, af_cv, af_pct)
    percent_drop = (1 - r) * 100.0
    temp_drop = percent_drop * deg_per_pct
    return base_temp - temp_drop

def required_fuel_rate_to_hold_heat(current_fuel_rate: float, coal_cv: float, af_cv: float, af_pct: float) -> float:
    r = heat_ratio(coal_cv, af_cv, af_pct)
    if r <= 0:
        return current_fuel_rate
    return current_fuel_rate / r
