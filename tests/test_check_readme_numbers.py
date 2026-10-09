"""README number check (T12): formatting helpers, and that the committed README agrees with results/."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_readme_numbers", ROOT / "scripts" / "check_readme_numbers.py")
crn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crn)


def test_formatting():
    assert crn.sg(0.325) == "+0.325" and crn.sg(-0.35) == "-0.350"
    assert crn.sg(-1e-17) == "+0.000" and crn.sg(0.0) == "+0.000"      # no "-0.000"
    assert crn.ci(0.245, 0.405) == "[+0.245, +0.405]"
    assert crn.fp(4.2113505e-13) == "4.2e-13" and crn.fp(3.1e-9) == "3.1e-9"
    assert crn.fp(0.0675) in ("0.067", "0.068") and crn.fp(0.002118) == "0.0021"
    assert crn.pct1(0.3015) == "30.1%"


def test_readme_agrees_with_results():
    ok, bad = crn.run_checks()
    assert not bad, bad
    assert len(ok) > 50
