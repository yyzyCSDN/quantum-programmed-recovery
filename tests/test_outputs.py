"""Behavior checks for the public instruction.md contract."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import cases
from checker import expected, verify
from runner import run_cases

@pytest.fixture(scope="session")
def results():
    return run_cases(cases(), "/app/photon_report.py", 4_000_000, expected, verify)

@pytest.mark.parametrize("case_id", ['quiet_ready'])
def test_quiet_ready(results, case_id):
    """instruction.md — Physical model: zero rates and initial ready state preserve the unnormalised joint state."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['initial_recovery_front'])
def test_initial_recovery_front(results, case_id):
    """instruction.md — Physical model: initial dead mass recovers at its own remaining time, before observation at that instant."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['pending_readout'])
def test_pending_readout(results, case_id):
    """instruction.md — Physical model: pending-to-readout latency follows bandwidth, independently of true photon emission."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['one_photon'])
def test_one_photon(results, case_id):
    """instruction.md — Input and output: emission and readout counts, covariance, and third cumulants remain distinct."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['unobserved_emission'])
def test_unobserved_emission(results, case_id):
    """instruction.md — Physical model: a true emission can remain unobserved while still incrementing N."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['blocked_electronics'])
def test_blocked_electronics(results, case_id):
    """instruction.md — Physical model: a zero electronic readout rate does not suppress true emissions."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['fixed_renewal'])
def test_fixed_renewal(results, case_id):
    """instruction.md — Physical model: fixed recovery waits retain history across successive readouts."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['first_readout_stopped'])
def test_first_readout_stopped(results, case_id):
    """instruction.md — Input and output: first_click.state stops after the first measurement and is not subsequently propagated."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['stretched_recovery'])
def test_stretched_recovery(results, case_id):
    """instruction.md — Physical model: positive dead_slope changes recovery time as a function of readout time."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['compressed_recovery'])
def test_compressed_recovery(results, case_id):
    """instruction.md — Physical model: negative dead_slope compresses the recovery-time distribution."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['projective_branch_memory'])
def test_projective_branch_memory(results, case_id):
    """instruction.md — Physical model: readout measurement branches preserve correlation between quantum state and recovery duration."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['weak_branch_feedback'])
def test_weak_branch_feedback(results, case_id):
    """instruction.md — Physical model: nonunitary generalised measurement operators include branch feedback."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['later_entries_return_first'])
def test_later_entries_return_first(results, case_id):
    """instruction.md — Physical model: recovery ordering need not match entry ordering."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['overlapping_mark_windows'])
def test_overlapping_mark_windows(results, case_id):
    """instruction.md — Physical model: simultaneous recovering mass is the sum of all valid branches and source segments."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['dead_quantum_transport'])
def test_dead_quantum_transport(results, case_id):
    """instruction.md — Physical model: quantum states in dead evolve under the actual intervening control history."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['dead_emission_tally'])
def test_dead_emission_tally(results, case_id):
    """instruction.md — Physical model: emissions during dead increment N without restarting recovery or incrementing C."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['initial_and_new_cohorts'])
def test_initial_and_new_cohorts(results, case_id):
    """instruction.md — Physical model: initial recovery and later measurement-generated recovery retain separate clocks."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['rare_first_readout'])
def test_rare_first_readout(results, case_id):
    """instruction.md — Input and output: no_click and unnormalised first-click quantities remain accurate for rare readouts."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['maximum_domain'])
def test_maximum_domain(results, case_id):
    """instruction.md — Range and output: three-dimensional, four-segment input preserves time order and all declared statistics."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

@pytest.mark.parametrize("case_id", ['random_instrument_00', 'random_instrument_01', 'random_instrument_02', 'random_instrument_03', 'random_instrument_04'])
def test_combined_instruments(results, case_id):
    """instruction.md — Physical model and range: combined valid measurement operators, control segments, and initial correlations."""
    row = results[case_id]
    assert row["returncode"] == 0, row["errors"]
    assert row["errors"] == [], "\n".join(row["errors"])

