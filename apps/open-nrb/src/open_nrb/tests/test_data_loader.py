#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
import math
import sys
import tempfile
from pathlib import Path

import torch

try:
    from open_nrb.models.data_loader import infer_sides, load_panel, read_rows
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from data_loader import infer_sides, load_panel, read_rows


_HEADER = (
    "source_file,year_bs,month_bs,page,statement,bfi_name,item_code,line_item,value_mn_npr\n"
)


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text(_HEADER + "".join(rows))


def _row(source_file, year_bs, month_bs, page, bfi, code, label, value) -> str:
    return (
        f"{source_file},{year_bs},{month_bs},{page},balance_sheet,{bfi},{code},{label},{value}\n"
    )


def _full_period_rows(
    source_file,
    year_bs,
    month_bs,
    bfi,
    *,
    liquid_funds,
    loans_advances,
    interbank_r,
    capital_fund,
    deposits,
    borrowings_nrb,
    borrowings_interbank,
    borrowings_total=None,
    extra_borrowing_subitems=None,
):
    """Build one complete, correctly-ordered bfi/period block: Liabilities
    section (with its TOTAL Liabilities marker) first, then Assets section
    (with its TOTAL ASSETS marker) second -- matching the real bulletin layout.
    All 10 real ASSET_TOP_LEVEL_LABELS rows are included (zero-filled unless given)
    so 'size' never silently breaks on a missing row. interbank_r is reported as
    "a. Interbank Lending", a sub-item folded into "3 SHARE & OTHER INVESTMENT"'s
    own total (matching real data, Sec 9.4) -- not a separate top-level row.
    """
    extra_borrowing_subitems = extra_borrowing_subitems or {}
    page = 1
    rows = [
        _row(source_file, year_bs, month_bs, page, bfi, "1", "1 CAPITAL FUND", capital_fund),
    ]
    if borrowings_total is not None:
        rows.append(
            _row(source_file, year_bs, month_bs, page, bfi, "2", "2 BORROWINGS", borrowings_total)
        )
    rows.append(_row(source_file, year_bs, month_bs, page, bfi, "a", "a. NRB", borrowings_nrb))
    rows.append(
        _row(
            source_file, year_bs, month_bs, page, bfi,
            "b", "b. Interbank Borrowing", borrowings_interbank,
        )
    )
    for code, label, value in [
        ("c", "c. Foreign Banks and Fin. Ins.", extra_borrowing_subitems.get("c")),
        ("d", "d. Other Financial Ins.", extra_borrowing_subitems.get("d")),
        ("e", "e. Bonds and Securities", extra_borrowing_subitems.get("e")),
    ]:
        if value is not None:
            rows.append(_row(source_file, year_bs, month_bs, page, bfi, code, label, value))
    if deposits is not None:
        rows.append(_row(source_file, year_bs, month_bs, page, bfi, "3", "3 DEPOSITS", deposits))
    rows.append(_row(source_file, year_bs, month_bs, page, bfi, "x", "TOTAL Liabilities", 0))

    asset_items = [
        ("1", "1 LIQUID FUNDS", liquid_funds),
        ("2", "2 INVESTMENT IN SECURITIES", 0),
        ("3", "3 SHARE & OTHER INVESTMENT", interbank_r),
        ("4", "4 LOANS & ADVANCES", loans_advances),
        ("5", "5 BILL PURCHASED", 0),
        ("6", "6 LOANS AGAINST COLLECTED BILLS", 0),
        ("7", "7 FIXED ASSETS", 0),
        ("8", "8 OTHER ASSETS", 0),
        ("9", "9 EXPENSES NOT WRITTEN OFF", 0),
        ("10", "10 NON BANKING ASSETS", 0),
    ]
    for code, label, value in asset_items:
        rows.append(_row(source_file, year_bs, month_bs, page, bfi, code, label, value))
    rows.append(
        _row(source_file, year_bs, month_bs, page, bfi, "a", "a. Interbank Lending", interbank_r)
    )
    rows.append(_row(source_file, year_bs, month_bs, page, bfi, "y", "TOTAL ASSETS", 0))
    return rows


def side_inference_collision_case() -> bool:
    """item_code=1 collides (CAPITAL FUND on liabilities, LIQUID FUNDS on assets);
    side must come from the TOTAL LIABILITIES / TOTAL ASSETS markers, not the code."""
    rows_csv = [
        _row("b.pdf", 2080, "Asar", 1, "NBL", "1", "1 CAPITAL FUND", 100),
        _row("b.pdf", 2080, "Asar", 1, "NBL", "2", "TOTAL Liabilities", 100),
        _row("b.pdf", 2080, "Asar", 1, "NBL", "1", "1 LIQUID FUNDS", 40),
        _row("b.pdf", 2080, "Asar", 1, "NBL", "3", "TOTAL ASSETS", 40),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        sides = infer_sides(read_rows(path))

    cap_side = sides.get("1 CAPITAL FUND")
    liq_side = sides.get("1 LIQUID FUNDS")
    ok = cap_side == "liabilities" and liq_side == "assets"
    print(
        f"[side_inference_collision_case] item_code=1 split correctly "
        f"capital_fund_side={cap_side} liquid_funds_side={liq_side} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def missing_marker_row_fallback_case() -> bool:
    """Pre-2075-era bulletins (confirmed against real Asar-2074.pdf, 2026-10-22) never
    extract a 'TOTAL Liabilities' marker row -- the liabilities section just ends and
    the assets section starts back at item_code '1' with no boundary cue. Side must
    fall back to the item_code restart itself (bare top-level codes strictly increase
    within one side), not require the marker row to exist."""
    rows_csv = [
        _row("i.pdf", 2074, "Asar", 1, "NBL", "1", "1 CAPITAL FUND", 10),
        _row("i.pdf", 2074, "Asar", 1, "NBL", "a.", "a. Paid-up Capital", 10),
        _row("i.pdf", 2074, "Asar", 1, "NBL", "5", "5 Other Liabilities", 5),
        _row("i.pdf", 2074, "Asar", 1, "NBL", "1.", "1. Sundry Creditors", 5),
        _row("i.pdf", 2074, "Asar", 1, "NBL", "1", "1 LIQUID FUNDS", 8),
        _row("i.pdf", 2074, "Asar", 1, "NBL", "2", "2 INVESTMENT IN SECURITIES", 7),
        _row("i.pdf", 2074, "Asar", 1, "NBL", "", "TOTAL ASSETS", 15),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        sides = infer_sides(read_rows(path))

    cap_side = sides.get("1 CAPITAL FUND")
    liq_side = sides.get("1 LIQUID FUNDS")
    ok = cap_side == "liabilities" and liq_side == "assets"
    print(
        f"[missing_marker_row_fallback_case] no-marker era: "
        f"capital_fund_side={cap_side} liquid_funds_side={liq_side} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def unwatched_label_collision_case() -> bool:
    """'e. Others' is reused across unrelated sections with opposite true sides
    (confirmed real data, Asar-2074.pdf/NBL, 2026-10-22: under '3 DEPOSITS' on
    liabilities, under '8 OTHER ASSETS' on assets). infer_sides must not raise on
    this -- it isn't one of the 8 fields _compute_fields actually reads -- while
    still correctly resolving the watched labels around it."""
    rows_csv = [
        _row("j.pdf", 2074, "Asar", 1, "NBL", "1", "1 CAPITAL FUND", 10),
        _row("j.pdf", 2074, "Asar", 1, "NBL", "3", "3 DEPOSITS", 20),
        _row("j.pdf", 2074, "Asar", 1, "NBL", "e.", "e. Others", 2),
        _row("j.pdf", 2074, "Asar", 1, "NBL", "", "TOTAL Liabilities", 30),
        _row("j.pdf", 2074, "Asar", 1, "NBL", "1", "1 LIQUID FUNDS", 8),
        _row("j.pdf", 2074, "Asar", 1, "NBL", "8", "8 OTHER ASSETS", 5),
        _row("j.pdf", 2074, "Asar", 1, "NBL", "e.", "e. Others", 3),
        _row("j.pdf", 2074, "Asar", 1, "NBL", "", "TOTAL ASSETS", 16),
    ]
    raised = False
    sides: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        try:
            sides = infer_sides(read_rows(path))
        except ValueError:
            raised = True

    cap_side = sides.get("1 CAPITAL FUND")
    liq_side = sides.get("1 LIQUID FUNDS")
    ok = not raised and cap_side == "liabilities" and liq_side == "assets"
    print(
        f"[unwatched_label_collision_case] raised={raised} "
        f"capital_fund_side={cap_side} liquid_funds_side={liq_side} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def covariate_formula_case() -> bool:
    rows_csv = _full_period_rows(
        "c.pdf", 2080, "Asar", "NBL",
        liquid_funds=40, loans_advances=50, interbank_r=10,
        capital_fund=20, deposits=25,
        borrowings_nrb=4, borrowings_interbank=6, borrowings_total=10,
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        panel = load_panel(path)

    x = panel.X[0, 0]
    # size = 40 (liquid) + 50 (loans) + 10 (interbank r) + zeros = 100
    expected = torch.tensor([100.0, 20 / 100, 40 / 100, 50 / 25, 4 / 10])
    finite_ok = torch.isfinite(x).all().item()
    close_ok = torch.allclose(x, expected, atol=1e-6)
    r_ok = panel.r[0, 0].item() == 10.0
    c_ok = panel.c[0, 0].item() == 6.0
    k_ok = panel.K[0, 0].item() == 20.0
    ok = finite_ok and close_ok and r_ok and c_ok and k_ok
    print(
        f"[covariate_formula_case] finite_ok={finite_ok} close_ok={close_ok} "
        f"r_ok={r_ok} c_ok={c_ok} k_ok={k_ok} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def optional_asset_category_blank_case() -> bool:
    """A single optional asset category with a blank value cell (e.g. a smaller BFI that
    doesn't report '9 EXPENSES NOT WRITTEN OFF') is dropped entirely by read_rows (Sec 9
    blank-value note), so it's simply absent from rec -- not a zero. 'size' must still sum
    the other 9 categories via _lookup_field_or_zero and the cell must load successfully,
    not raise ValueError / drop the whole bfi-period cell the way strict _lookup_field did."""
    rows_csv = _full_period_rows(
        "k.pdf", 2080, "Asar", "NBL",
        liquid_funds=40, loans_advances=50, interbank_r=10,
        capital_fund=20, deposits=25,
        borrowings_nrb=4, borrowings_interbank=6, borrowings_total=10,
    )
    rows_csv = [
        row.rsplit(",", 1)[0] + ",\n" if "9 EXPENSES NOT WRITTEN OFF" in row else row
        for row in rows_csv
    ]
    raised = False
    panel = None
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        try:
            panel = load_panel(path)
        except ValueError:
            raised = True

    ok = not raised
    if ok:
        x = panel.X[0, 0]
        # size = 40 (liquid) + 50 (loans) + 10 (interbank r) + zeros; blank "9 ..." -> 0,
        # not KeyError
        expected = torch.tensor([100.0, 20 / 100, 40 / 100, 50 / 25, 4 / 10])
        ok = torch.isfinite(x).all().item() and torch.allclose(x, expected, atol=1e-6)
    print(f"[optional_asset_category_blank_case] raised={raised} -> {'PASS' if ok else 'FAIL'}")
    return ok


def borrowings_fallback_case() -> bool:
    """When '2 BORROWINGS' subtotal is unreported, denominator falls back to the
    sum of confirmed real sub-items a.-e. (Sec 8b/9.4 note on X^(5))."""
    rows_csv = _full_period_rows(
        "d.pdf", 2075, "Asar", "NBL",
        liquid_funds=10, loans_advances=10, interbank_r=5,
        capital_fund=5, deposits=10,
        borrowings_nrb=2, borrowings_interbank=3, borrowings_total=None,
        extra_borrowing_subitems={"c": 1},
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        panel = load_panel(path)

    # borrowings_total fallback = 2 (nrb) + 3 (interbank) + 1 (c) = 6; X^(5) = 2/6
    ok = math.isclose(panel.X[0, 0, 4].item(), 2 / 6, abs_tol=1e-6)
    print(
        f"[borrowings_fallback_case] nrb_dependence={panel.X[0, 0, 4].item():.6f} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def period_chronology_rollover_case() -> bool:
    """Chaitra (month 12) of year Y must sort before Baisakh (month 1) of year
    Y+1 -- lexicographic month-name sort would get this wrong."""
    rows_csv = _full_period_rows(
        "e.pdf", 2079, "Chaitra", "NBL",
        liquid_funds=1, loans_advances=1, interbank_r=1,
        capital_fund=1, deposits=1,
        borrowings_nrb=1, borrowings_interbank=1, borrowings_total=1,
    ) + _full_period_rows(
        "e.pdf", 2080, "Baisakh", "NBL",
        liquid_funds=1, loans_advances=1, interbank_r=1,
        capital_fund=1, deposits=1,
        borrowings_nrb=1, borrowings_interbank=1, borrowings_total=1,
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        panel = load_panel(path)

    ok = panel.periods == [(2079, "Chaitra"), (2080, "Baisakh")]
    print(
        f"[period_chronology_rollover_case] periods={panel.periods} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def missing_field_left_nan_case() -> bool:
    """A cell missing a required field entirely (e.g. no DEPOSITS row at all) is a real
    extraction gap, confirmed present even among institutions later dropped by
    filter_commercial_banks (2026-10-25) -- load_panel builds a dense panel for every raw
    bfi_name, not just whatever subset a caller filters to afterward, so this must load
    successfully with the cell left NaN (same convention as zero_deposits_left_nan_case),
    not kill the whole panel load the way it used to."""
    rows_csv = _full_period_rows(
        "f.pdf", 2080, "Asar", "NBL",
        liquid_funds=1, loans_advances=1, interbank_r=1,
        capital_fund=1, deposits=None,  # omitted on purpose
        borrowings_nrb=1, borrowings_interbank=1, borrowings_total=1,
    )
    raised = False
    panel = None
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        try:
            panel = load_panel(path)
        except ValueError:
            raised = True

    ok = not raised and panel.X[0, 0].isnan().all().item()
    print(f"[missing_field_left_nan_case] raised={raised} -> {'PASS' if ok else 'FAIL'}")
    return ok


def omitted_nrb_borrowing_case() -> bool:
    """'a. NRB' omitted entirely (not printed as 0), confirmed live in real data (1546 cells,
    e.g. ADBNL across many periods) for a bank/period with no NRB borrowing -- same
    leaf-omission convention as the optional asset categories above (borrowings_nrb is now
    looked up via _lookup_field_or_zero, not the strict _lookup_field). Must load successfully
    with nrb_dependence computed as 0, not raise."""
    rows_csv = _full_period_rows(
        "onb.pdf", 2080, "Asar", "NBL",
        liquid_funds=10, loans_advances=10, interbank_r=5,
        capital_fund=5, deposits=10,
        borrowings_nrb=0, borrowings_interbank=3, borrowings_total=10,
    )
    rows_csv = [row for row in rows_csv if "a. NRB" not in row]
    raised = False
    panel = None
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        try:
            panel = load_panel(path)
        except ValueError:
            raised = True

    ok = not raised
    if ok:
        nrb_dependence = panel.X[0, 0, 4].item()
        ok = math.isclose(nrb_dependence, 0.0, abs_tol=1e-9)
    print(f"[omitted_nrb_borrowing_case] raised={raised} -> {'PASS' if ok else 'FAIL'}")
    return ok


def zero_borrowings_total_case() -> bool:
    """borrowings_total==0 means NO borrowing of any kind (confirmed live in real data, caused
    a ZeroDivisionError) -- borrowings_nrb is a strict subset of borrowings_total, so this is
    legitimately 0 NRB dependence, not missing data; the cell must load successfully."""
    rows_csv = _full_period_rows(
        "zbt.pdf", 2080, "Asar", "NBL",
        liquid_funds=10, loans_advances=10, interbank_r=5,
        capital_fund=5, deposits=10,
        borrowings_nrb=0, borrowings_interbank=0, borrowings_total=0,
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        panel = load_panel(path)

    nrb_dependence = panel.X[0, 0, 4].item()
    ok = math.isclose(nrb_dependence, 0.0, abs_tol=1e-9)
    print(
        f"[zero_borrowings_total_case] nrb_dependence={nrb_dependence} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def zero_deposits_left_nan_case() -> bool:
    """deposits==0 (e.g. a non-deposit-taking BFI, confirmed live in real data) has no
    principled ldr substitute value -- unlike zero_borrowings_total_case, this must leave the
    cell NaN (same convention as a bank with zero rows that period), NOT raise/kill the whole
    load the way a genuinely missing field does."""
    rows_csv = _full_period_rows(
        "zd.pdf", 2080, "Asar", "NBL",
        liquid_funds=10, loans_advances=10, interbank_r=5,
        capital_fund=5, deposits=0,
        borrowings_nrb=2, borrowings_interbank=3, borrowings_total=10,
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        panel = load_panel(path)

    ok = panel.X[0, 0].isnan().all().item()
    print(f"[zero_deposits_left_nan_case] all_nan={ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def truncated_label_matching_case() -> bool:
    """Mid-word label truncation seen in some real extractions (e.g. '1 CAPITAL FUND' ->
    '1 CAPITAL', '4 LOANS & ADVANCES' -> '4 LOANS &') must still resolve to the right
    field via mutual-prefix matching, not raise/misfire (Sec 9.5)."""
    rows_csv = _full_period_rows(
        "g.pdf", 2080, "Asar", "NBL",
        liquid_funds=10, loans_advances=20, interbank_r=5,
        capital_fund=8, deposits=16,
        borrowings_nrb=2, borrowings_interbank=3, borrowings_total=10,
    )
    rows_csv = [
        row.replace("1 CAPITAL FUND", "1 CAPITAL").replace("4 LOANS & ADVANCES", "4 LOANS &")
        for row in rows_csv
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        panel = load_panel(path)

    k_ok = panel.K[0, 0].item() == 8.0
    ldr_ok = math.isclose(panel.X[0, 0, 3].item(), 20 / 16, abs_tol=1e-6)
    ok = k_ok and ldr_ok
    print(
        f"[truncated_label_matching_case] k_ok={k_ok} ldr_ok={ldr_ok} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def multi_bank_interleaved_case() -> bool:
    """Real bulletins extract row-major by line item -- bank A's row for item N is
    immediately followed by bank B's row for item N, not bank A's whole sheet then
    bank B's whole sheet (confirmed against real data 2026-10-22, caused a live
    'inconsistent side' ValueError). infer_sides must key its liabilities->assets
    state machine per (source_file, bfi_name), not just per file, or bank B's
    TOTAL LIABILITIES marker collides with bank A's already-flipped state."""
    rows_csv = [
        _row("h.pdf", 2080, "Asar", 1, "NBL", "1", "1 CAPITAL FUND", 10),
        _row("h.pdf", 2080, "Asar", 1, "EBL", "1", "1 CAPITAL FUND", 20),
        _row("h.pdf", 2080, "Asar", 1, "NBL", "x", "TOTAL Liabilities", 10),
        _row("h.pdf", 2080, "Asar", 1, "EBL", "x", "TOTAL Liabilities", 20),
        _row("h.pdf", 2080, "Asar", 1, "NBL", "1", "1 LIQUID FUNDS", 5),
        _row("h.pdf", 2080, "Asar", 1, "EBL", "1", "1 LIQUID FUNDS", 15),
        _row("h.pdf", 2080, "Asar", 1, "NBL", "y", "TOTAL ASSETS", 5),
        _row("h.pdf", 2080, "Asar", 1, "EBL", "y", "TOTAL ASSETS", 15),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.csv"
        _write_csv(path, rows_csv)
        sides = infer_sides(read_rows(path))

    cap_side = sides.get("1 CAPITAL FUND")
    liq_side = sides.get("1 LIQUID FUNDS")
    ok = cap_side == "liabilities" and liq_side == "assets"
    print(
        f"[multi_bank_interleaved_case] capital_fund_side={cap_side} "
        f"liquid_funds_side={liq_side} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


CASES = [
    side_inference_collision_case,
    missing_marker_row_fallback_case,
    unwatched_label_collision_case,
    multi_bank_interleaved_case,
    covariate_formula_case,
    optional_asset_category_blank_case,
    borrowings_fallback_case,
    period_chronology_rollover_case,
    missing_field_left_nan_case,
    omitted_nrb_borrowing_case,
    zero_borrowings_total_case,
    zero_deposits_left_nan_case,
    truncated_label_matching_case,
]


def test_side_inference_collision():
    assert side_inference_collision_case()


def test_missing_marker_row_fallback():
    assert missing_marker_row_fallback_case()


def test_unwatched_label_collision():
    assert unwatched_label_collision_case()


def test_multi_bank_interleaved():
    assert multi_bank_interleaved_case()


def test_covariate_formula():
    assert covariate_formula_case()


def test_optional_asset_category_blank():
    assert optional_asset_category_blank_case()


def test_borrowings_fallback():
    assert borrowings_fallback_case()


def test_period_chronology_rollover():
    assert period_chronology_rollover_case()


def test_missing_field_left_nan():
    assert missing_field_left_nan_case()


def test_omitted_nrb_borrowing():
    assert omitted_nrb_borrowing_case()


def test_zero_borrowings_total():
    assert zero_borrowings_total_case()


def test_zero_deposits_left_nan():
    assert zero_deposits_left_nan_case()


def test_truncated_label_matching():
    assert truncated_label_matching_case()


if __name__ == "__main__":
    results = {case.__name__: case() for case in CASES}
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if all(results.values()) else 'FAIL'}")
