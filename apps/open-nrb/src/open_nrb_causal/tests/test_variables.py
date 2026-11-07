#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
"""Tests for outcome/dose variable construction from NRB P&L data (variables.py)."""

import sys
import tempfile
from pathlib import Path

import torch

try:
    from open_nrb.models.data_loader import COVARIATE_ORDER, Panel
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper1_model"))
    from models.data_loader import COVARIATE_ORDER, Panel  # type: ignore[no-redef]

try:
    from open_nrb_causal.models.estimator import fit_did
    from open_nrb_causal.models.variables import (
        CAPITAL_FLOOR,
        check_pnl_csv_exists,
        compute_dose,
        controls_from_panel,
        load_outcome_panel,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from estimator import fit_did  # type: ignore[no-redef]
    from variables import (  # type: ignore[no-redef]
        CAPITAL_FLOOR,
        check_pnl_csv_exists,
        compute_dose,
        controls_from_panel,
        load_outcome_panel,
    )

_PNL_HEADER = "source_file,year_bs,month_bs,page,statement,bfi_name,item_code,line_item,value_mn_npr\n"


def _pnl_row(bank: str, month: str, item_code: str, line_item: str, value: float) -> str:
    """Format one NRB profit-and-loss CSV row for the given bank/month/item/value."""
    return f"Asar-2074.pdf,2074,{month},25,profit_and_loss,{bank},{item_code},{line_item},{value}\n"


def _outcome_panel_fixture() -> Panel:
    """Build a 2-bank, 2-period Panel fixture for outcome-loading tests."""
    n, t = 2, 2
    return Panel(
        bfi_names=["NBL", "RBB"],
        periods=[(2074, "Asar"), (2074, "Shrawan")],
        X=torch.zeros(n, t, len(COVARIATE_ORDER)),
        r=torch.zeros(n, t),
        c=torch.zeros(n, t),
        K=torch.zeros(n, t),
    )


def _real_schema_pnl_csv() -> str:
    """Return a real-schema P&L CSV covering both panel banks across the needed line items."""
    return _PNL_HEADER + "".join([
        _pnl_row("NBL", "Asar", "1.", "1. Interest Expense", 100.0),
        _pnl_row("NBL", "Asar", "", "1.1 On Deposit Liabilities", 80.0),
        _pnl_row("NBL", "Asar", "1.", "1. Interest Income", 150.0),
        _pnl_row("NBL", "Asar", "", "7.1.1 General Loan Loss Provision", 5.0),
        _pnl_row("NBL", "Asar", "", "7.1.2 Special Loan Loss Provision", 3.0),
        _pnl_row("NBL", "Asar", "", "7.1.3 Additional Loan Loss Provision", 2.0),
        _pnl_row("NBL", "Asar", "8.", "8. Loan Written Off", 1.0),
        _pnl_row("NBL", "Asar", "12.", "12. Net Profit", 20.0),
        _pnl_row("RBB", "Asar", "1.", "1. Interest Expense", 50.0),
        _pnl_row("RBB", "Asar", "1.", "1. Interest Income", 40.0),
        _pnl_row("RBB", "Asar", "", "7.1.1 General Loan Loss Provision", 1.0),
        _pnl_row("RBB", "Asar", "8.", "8. Loan Written Off", 0.0),
        _pnl_row("RBB", "Asar", "9.", "9. Net Loss", 7.0),
        _pnl_row("NBL", "Shrawan", "1.", "1. Interest Expense", 10.0),
        _pnl_row("NBL", "Shrawan", "1.", "1. Interest Income", 12.0),
        _pnl_row("NOTINPANEL", "Asar", "1.", "1. Interest Expense", 999.0),
    ])


def load_outcome_panel_real_schema_case() -> bool:
    """Assert load_outcome_panel builds outcomes from the real NRB P&L schema.

    Checks the accounting construction per bank-period: nim = interest income -
    interest expense, loan_loss_provision = general + special + additional
    provisions, net_profit_loss is positive for a net profit and negative for a
    net loss, and loan_written_off is read directly; banks not in the panel and
    months with no rows stay NaN.
    """
    nan = float("nan")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nrb_profit_and_loss.csv"
        path.write_text(_real_schema_pnl_csv())
        out = load_outcome_panel(path, _outcome_panel_fixture())
    expected = {
        "nim": torch.tensor([[50.0, 2.0], [-10.0, nan]]),
        "loan_loss_provision": torch.tensor([[10.0, nan], [1.0, nan]]),
        "net_profit_loss": torch.tensor([[20.0, nan], [-7.0, nan]]),
        "loan_written_off": torch.tensor([[1.0, nan], [0.0, nan]]),
    }
    passed = all(
        torch.allclose(getattr(out, name), want, equal_nan=True) for name, want in expected.items()
    )
    print(f"[load_outcome_panel_real_schema_case] "
          f"{ {name: getattr(out, name).tolist() for name in expected} } "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def load_outcome_panel_raises_on_conflicting_rows_case() -> bool:
    """Assert load_outcome_panel raises ValueError on conflicting duplicate rows.

    A second row for the same bank/month/item with a different value makes the
    loader raise ValueError mentioning "more than one distinct".
    """
    csv_text = _real_schema_pnl_csv() + _pnl_row("NBL", "Asar", "1.", "1. Interest Expense", 101.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nrb_profit_and_loss.csv"
        path.write_text(csv_text)
        try:
            load_outcome_panel(path, _outcome_panel_fixture())
            passed = False
        except ValueError as e:
            passed = "more than one distinct" in str(e)
    print(f"[load_outcome_panel_raises_on_conflicting_rows_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def load_outcome_panel_raises_when_outcome_has_no_data_case() -> bool:
    """Assert load_outcome_panel raises ValueError when an outcome has no data.

    A CSV carrying only an interest-expense row (so nim cannot be built) makes
    the loader raise ValueError mentioning "no data" and the uppercased line
    item "1. INTEREST EXPENSE".
    """
    csv_text = _PNL_HEADER + _pnl_row("NBL", "Asar", "1.", "1. Interest Expense", 100.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nrb_profit_and_loss.csv"
        path.write_text(csv_text)
        try:
            load_outcome_panel(path, _outcome_panel_fixture())
            passed = False
        except ValueError as e:
            passed = "no data" in str(e) and "1. INTEREST EXPENSE" in str(e)
    print(f"[load_outcome_panel_raises_when_outcome_has_no_data_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def fit_did_ignores_nan_outcome_cells_case() -> bool:
    """Assert fit_did ignores NaN outcome cells while recovering the true betas.

    Two NaN outcome cells are dropped (n_obs == n*t - 2) and fit_did still
    recovers beta_dose ~ 2.0 and beta_exposure ~ -1.0.
    """
    torch.manual_seed(0)
    n, t = 6, 8
    dose = torch.randn(n, t)
    exposure = torch.randn(n, t)
    y = 2.0 * dose - 1.0 * exposure + torch.randn(n, 1) + torch.randn(1, t)
    y_holey = y.clone()
    y_holey[0, 0] = float("nan")
    y_holey[3, 5] = float("nan")
    res = fit_did(y_holey, dose, exposure, {}, torch.ones(n, t))
    passed = (
        res.n_obs == n * t - 2
        and abs(res.beta_dose - 2.0) < 1e-3
        and abs(res.beta_exposure + 1.0) < 1e-3
    )
    print(f"[fit_did_ignores_nan_outcome_cells_case] n_obs={res.n_obs} beta_dose={res.beta_dose:.4f} "
          f"beta_exposure={res.beta_exposure:.4f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def _make_panel() -> Panel:
    """Build a 2-bank, 1-period Panel fixture with known covariate-column values."""
    n, t = 2, 1
    X = torch.zeros(n, t, len(COVARIATE_ORDER))
    X[0, 0, COVARIATE_ORDER.index("capital_ratio")] = 0.08
    X[1, 0, COVARIATE_ORDER.index("capital_ratio")] = 0.15
    X[:, :, COVARIATE_ORDER.index("liquidity_ratio")] = 0.3
    X[:, :, COVARIATE_ORDER.index("ldr")] = 0.8
    X[:, :, COVARIATE_ORDER.index("nrb_dependence")] = 0.05
    return Panel(
        bfi_names=["A", "B"],
        periods=[(2080, "Asar")],
        X=X,
        r=torch.rand(n, t),
        c=torch.rand(n, t),
        K=torch.rand(n, t),
    )


def compute_dose_matches_formula_case() -> bool:
    """Assert compute_dose equals capital_floor minus the bank's capital ratio.

    dose = capital_floor - capital_ratio elementwise (0.11 - 0.08, 0.11 - 0.15),
    matched to atol 1e-6.
    """
    panel = _make_panel()
    capital_floor = 0.11
    dose = compute_dose(panel, capital_floor)
    expected = torch.tensor([[0.11 - 0.08], [0.11 - 0.15]])
    passed = torch.allclose(dose, expected, atol=1e-6)
    print(f"[compute_dose_matches_formula_case] dose={dose.flatten().tolist()} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def check_pnl_csv_exists_raises_on_missing_file_case() -> bool:
    """Assert check_pnl_csv_exists raises FileNotFoundError for a missing file."""
    missing_path = Path("/tmp/definitely_does_not_exist_open_nrb_causal_test.csv")
    try:
        check_pnl_csv_exists(missing_path)
        passed = False
    except FileNotFoundError:
        passed = True
    print(f"[check_pnl_csv_exists_raises_on_missing_file_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def check_pnl_csv_exists_passes_on_real_file_case() -> bool:
    """Assert check_pnl_csv_exists passes for an existing file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pnl.csv"
        path.write_text("source_file,bfi_name\n")
        try:
            check_pnl_csv_exists(path)
            passed = True
        except FileNotFoundError:
            passed = False
    print(f"[check_pnl_csv_exists_passes_on_real_file_case] -> {'PASS' if passed else 'FAIL'}")
    return passed


def controls_from_panel_pulls_right_columns_case() -> bool:
    """Assert controls_from_panel pulls the liquidity_ratio, ldr, and nrb_dependence columns.

    The returned controls have exactly those three keys, holding the panel's
    column values (0.3, 0.8, 0.05).
    """
    panel = _make_panel()
    controls = controls_from_panel(panel)
    passed = (
        set(controls.keys()) == {"liquidity_ratio", "ldr", "nrb_dependence"}
        and torch.allclose(controls["liquidity_ratio"], torch.full((2, 1), 0.3))
        and torch.allclose(controls["ldr"], torch.full((2, 1), 0.8))
        and torch.allclose(controls["nrb_dependence"], torch.full((2, 1), 0.05))
    )
    print(f"[controls_from_panel_pulls_right_columns_case] keys={list(controls.keys())} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def capital_floor_is_the_verified_leverage_ratio_case() -> bool:
    """Assert CAPITAL_FLOOR stays the verified NRB leverage ratio of 0.04.

    Guards against CAPITAL_FLOOR silently reverting to the None placeholder or
    drifting from the verified NRB leverage-ratio figure (variables.py's
    docstring: Capital Adequacy Framework 2015, Sec 2.7.B.a): it must be a float
    equal to 0.04.
    """
    passed = isinstance(CAPITAL_FLOOR, float) and abs(CAPITAL_FLOOR - 0.04) < 1e-9
    print(f"[capital_floor_is_the_verified_leverage_ratio_case] CAPITAL_FLOOR={CAPITAL_FLOOR} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


CASES = [
    compute_dose_matches_formula_case,
    check_pnl_csv_exists_raises_on_missing_file_case,
    check_pnl_csv_exists_passes_on_real_file_case,
    controls_from_panel_pulls_right_columns_case,
    capital_floor_is_the_verified_leverage_ratio_case,
    load_outcome_panel_real_schema_case,
    load_outcome_panel_raises_on_conflicting_rows_case,
    load_outcome_panel_raises_when_outcome_has_no_data_case,
    fit_did_ignores_nan_outcome_cells_case,
]


def test_load_outcome_panel_real_schema():
    """Assert load_outcome_panel_real_schema passes."""
    assert load_outcome_panel_real_schema_case()


def test_load_outcome_panel_raises_on_conflicting_rows():
    """Assert load_outcome_panel_raises_on_conflicting_rows passes."""
    assert load_outcome_panel_raises_on_conflicting_rows_case()


def test_load_outcome_panel_raises_when_outcome_has_no_data():
    """Assert load_outcome_panel_raises_when_outcome_has_no_data passes."""
    assert load_outcome_panel_raises_when_outcome_has_no_data_case()


def test_fit_did_ignores_nan_outcome_cells():
    """Assert fit_did_ignores_nan_outcome_cells passes."""
    assert fit_did_ignores_nan_outcome_cells_case()


def test_compute_dose_matches_formula():
    """Assert compute_dose_matches_formula passes."""
    assert compute_dose_matches_formula_case()


def test_capital_floor_is_the_verified_leverage_ratio():
    """Assert capital_floor_is_the_verified_leverage_ratio passes."""
    assert capital_floor_is_the_verified_leverage_ratio_case()


def test_check_pnl_csv_exists_raises_on_missing_file():
    """Assert check_pnl_csv_exists_raises_on_missing_file passes."""
    assert check_pnl_csv_exists_raises_on_missing_file_case()


def test_check_pnl_csv_exists_passes_on_real_file():
    """Assert check_pnl_csv_exists_passes_on_real_file passes."""
    assert check_pnl_csv_exists_passes_on_real_file_case()


def test_controls_from_panel_pulls_right_columns():
    """Assert controls_from_panel_pulls_right_columns passes."""
    assert controls_from_panel_pulls_right_columns_case()


if __name__ == "__main__":
    results = {case.__name__: case() for case in CASES}
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
