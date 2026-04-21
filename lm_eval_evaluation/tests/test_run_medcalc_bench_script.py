from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_medcalc_bench.py"
SPEC = spec_from_file_location("run_medcalc_bench_script", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
script = module_from_spec(SPEC)
SPEC.loader.exec_module(script)


def test_normalize_method_name_accepts_uncertainty_gate():
    assert (
        script.normalize_method_name("medcalc_uncertainty_consistency_gate")
        == "medcalc_uncertainty_consistency_gate"
    )


def test_split_to_task_name_still_routes_prompt_only_methods():
    assert script.split_to_task_name("test", "zero_shot_cot") == "med_calc_bench_zero_shot_cot"
