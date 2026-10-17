import re
import pandas as pd

# ---------------------------------------------------------------------------
# Commercial bank identity: short code -> full name (per NRB's published
# "Name list of Bank and Financial Institutions" across several years), plus
# a merger/panel-identity map for codes that change mid-series.
# ---------------------------------------------------------------------------

COMMERCIAL_BANKS = {
    'NBL': 'Nepal Bank Limited',
    'RBB': 'Rastriya Banijya Bank',
    'NABIL': 'Nabil Bank Limited',
    'NIBL': 'Nepal Investment Bank Limited',            # pre-merger code; see _PANEL_ENTITY_MAP
    'NIMB': 'Nepal Investment Mega Bank Limited',       # post-merger (NIBL + Mega), ~2080 BS onward
    'SCBNL': 'Standard Chartered Bank Nepal Limited',
    'HBL': 'Himalayan Bank Limited',
    'NSBI': 'Nepal SBI Bank Limited',
    'NBB': 'Nepal Bangaladesh Bank Limited',            # also seen spelled 'NBBL' -- verify against raw data
    'NBBL': 'Nepal Bangaladesh Bank Limited',
    'EBL': 'Everest Bank Limited',
    'BOK': 'Bank of Kathmandu Limited',                 # later "...Lumbini Limited" after absorbing Lumbini Bank
    'NCC': 'Nepal Credit and Commerce Bank Limited',
    'NIC': 'NIC Asia Bank Limited',                     # formerly "Nepal Industrial and Commercial Bank Limited" before merging with Bank of Asia
    'Lumbini': 'Lumbini Bank Limited',                  # absorbed into BOK
    'MBL': 'Machhapuchhre Bank Limited',
    'Kumari': 'Kumari Bank Limited',
    'Laxmi': 'Laxmi Sunrise Bank Limited',              # formerly "Laxmi Bank Limited" before merging with Sunrise Bank
    'SBL': 'Siddhartha Bank Limited',
    'ADBNL': 'Agriculture Development Bank',
    'Global': 'Global IME Bank Limited',                # formerly "Global Bank Limited" before merging with IME
    'Citizens': 'Citizens Bank International Limited',  # also seen spelled 'Citizen' -- verify against raw data
    'Citizen': 'Citizens Bank International Limited',
    'Prime': 'Prime Commercial Bank Limited',
    'BOA': 'Bank of Asia Nepal Limited',                # absorbed into NIC -> NIC Asia
    'Sunrise': 'Sunrise Bank Limited',                  # absorbed into Laxmi -> Laxmi Sunrise
    'DCBL': 'DCBL Bank Limited',
    'NMB': 'NMB Bank Limited',
    'Kist': 'Kist Bank Limited',                        # folded into Prabhu Bank's 2016 formation
    'Janata': 'Janata Bank Nepal Limited',
    'Mega': 'Mega Bank Nepal Limited',                  # absorbed into NIBL -> NIMB
    'CTBNL': 'Commerz and Trust Bank Nepal Limited',
    'Civil': 'Civil Bank Limited',                      # also seen spelled 'CBL' -- verify against raw data
    'CBL': 'Civil Bank Limited',
    'Century': 'Century Commercial Bank Limited',
    'Sanima': 'Sanima Bank Limited',
    'Prabhu': 'Prabhu Bank Limited',                    # also seen spelled 'PRABHU' -- verify against raw data
    'PRABHU': 'Prabhu Bank Limited',
}

# Codes that are literally different strings in the raw PDFs but refer to the
# same continuous economic entity across a merger. Without this, the pivot
# shows a hole in one code's year range right where the other code's range
# begins -- it LOOKS like a missing-data gap but is actually just two names
# for one bank, and it silently breaks panel/fixed-effects continuity.
_PANEL_ENTITY_MAP = {
    'NIBL': 'NIBL_NIMB', 'NIMB': 'NIBL_NIMB',
    'Laxmi': 'LAXMI_SUNRISE', 'Sunrise': 'LAXMI_SUNRISE',
    'BOK': 'BOK_LUMBINI', 'Lumbini': 'BOK_LUMBINI',
    'NIC': 'NIC_BOA', 'BOA': 'NIC_BOA',
}

# Raw spelling variants that are the same code, not a merger -- collapse
# before applying _PANEL_ENTITY_MAP.
_SPELLING_ALIASES = {'NBBL': 'NBB', 'Citizen': 'Citizens', 'CBL': 'Civil', 'PRABHU': 'Prabhu'}

# Bare fragment wordings that are the same concept across fiscal-year
# templates (e.g. "Domestic" vs "Domestic Currency") — merge to the bare form
# so they land in one pivot column instead of partitioning rows by era.
_FRAGMENT_ALIASES = {
    'domestic currency': 'domestic',
    'foreign currency': 'foreign',
    'private sector': 'private',
    'financial institutions': 'financial',
    'government organizations': 'government',
    # Confirmed via find_candidate_duplicate_columns: 0% overlap with
    # 'government' under the same parent ('8 OTHER ASSETS') — era wording
    # drift, not a distinct category.
    'government enterprises': 'government',
}


# Top-level canonical labels that resolve_ambiguous_fragments() watches for
# while walking rows in order, to know which section an AMBIGUOUS_FRAGMENT
# row belongs to.
_TOP_LEVEL_LABELS = {
    '1 CAPITAL FUND', '1 LIQUID FUNDS', '2 BORROWINGS', '2 INVESTMENT IN SECURITIES',
    '3 DEPOSITS', '3 SHARE & OTHER INVESTMENT', '4 LOANS & ADVANCES', '4 BILLS PAYABLE',
    '5 OTHER LIABILITIES',
    '5 BILL PURCHASED', '6 LOANS AGAINST COLLECTED BILLS', '7 FIXED ASSETS',
    '8 OTHER ASSETS', '9 EXPENSES NOT WRITTEN OFF', '10 NON BANKING ASSETS',
    '6 RECONCILIATION ACCOUNT', '7 PROFIT & LOSS A/C', '8 CALLS IN ADVANCE',
}

# Paper 1 Size (X^(1)): every asset-side top-level column, so Size tracks true
# total assets regardless of which of these later get excluded as standalone
# covariates via legacy_cols. '6 RECONCILIATION ACCOUNT' is deliberately left
# out -- it's a plug/adjustment item that can sit on either side of the sheet
# and isn't economically meaningful (Paper 1 §8d), so including it in a sum
# would add noise rather than real asset value.
ASSET_TOP_LEVEL = {
    '1 LIQUID FUNDS', '2 INVESTMENT IN SECURITIES', '3 SHARE & OTHER INVESTMENT',
    '4 LOANS & ADVANCES', '5 BILL PURCHASED', '6 LOANS AGAINST COLLECTED BILLS',
    '7 FIXED ASSETS', '8 OTHER ASSETS', '9 EXPENSES NOT WRITTEN OFF',
    '10 NON BANKING ASSETS', 'INTERBANK LENDING (Marginal r_i)',
}

# Some early-era filings never print an explicit '2 BORROWINGS' subtotal row --
# only its components (e.g. '2.a Borrowings - NRB', '4. In Foreign Banks') are
# present. Reconstruct the total from whichever sub-items exist rather than
# treating nrb_dependence as unrecoverable for those rows.
BORROWINGS_SUBITEMS = {
    '2.a Borrowings - NRB', '2.b Interbank Borrowing',
    '2.c Foreign Banks and Financial Institutions', '2.d Other Financial Institutions',
    '2.e Bonds and Securities / NRB Bonds',
    '2.f Borrowings - Govt Non-Financial Institutions',
    '2.g Borrowings - Other Non-Financial Institutions',
}

# Bare 'sundry' fragments that should land on the SAME explicit label as their
# fully-spelled sibling ('Sundry Creditors' / 'Sundry Debtors'), not a separate
# 'PARENT > SUNDRY' bucket — confirmed via null-count mirroring (510 null on
# the ambiguous side vs. 2 null on the explicit side, and vice versa for
# Other Assets): a handful of filings truncate the raw text to bare "Sundry".
_SUNDRY_PARENT_OVERRIDES = {
    '5 OTHER LIABILITIES': '5.1 Sundry Creditors',
    '8 OTHER ASSETS': '8.b Other Assets - Sundry Debtors',
}

# "1. In Nepal Rastra" (truncated, missing "Bank") wraps to the same text
# whether it's a Borrowings-NRB row or a Liquid-Funds Bank-Balance-NRB row --
# disambiguate by which top-level section it's actually under.
_NEPAL_RASTRA_PARENT_OVERRIDES = {
    '2 BORROWINGS': '2.a Borrowings - NRB',
    '1 LIQUID FUNDS': '1.b.1 Bank Balance - NRB',
}


def normalize_bfi_name(name: str) -> str:
    """Collapse spelling variants and merger-chain codes to one canonical
    panel id. Apply AFTER filtering to COMMERCIAL_BANKS and BEFORE pivoting.
    """
    if not isinstance(name, str):
        return name
    canon = _SPELLING_ALIASES.get(name, name)
    return _PANEL_ENTITY_MAP.get(canon, canon)


def canonicalize_line_item_regex(item: str) -> str:
    if not isinstance(item, str):
        return "UNKNOWN"

    s = item.strip()
    s_lower = s.lower()

    if s_lower in ['.', 'c.', 'e.', 'a.', 'b.', '4.', 'assets', 'liabilities']:
        return 'DROP_FRAGMENT'
    if re.search(r'\btotal\s+(assets|liabilities)\b', s_lower):
        return 'DROP_TOTAL_ROW'
    if s_lower.startswith('statement of assets and liabilities'):
        return 'DROP_FRAGMENT'

    # Specific multi-word headers must be checked before any bare single-word
    # pattern that could shadow them (e.g. "fixed assets" before bare "fixed").
    if re.search(r'\bcapital\s+fund\b', s_lower):
        return '1 CAPITAL FUND'
    elif re.search(r'\bliquid\s+funds?\b', s_lower):
        return '1 LIQUID FUNDS'
    elif re.search(r'\bborrowings?\b', s_lower) and not re.search(r'\binterbank\b', s_lower):
        return '2 BORROWINGS'
    elif re.search(r'\binvestment\s+in\s+securities\b', s_lower):
        return '2 INVESTMENT IN SECURITIES'
    elif re.search(r'\bdeposits?\b', s_lower) and not re.search(r'\bcall\b', s_lower):
        return '3 DEPOSITS'
    elif re.search(r'\bshare\s*&\s*other(\s+investment)?\b', s_lower):
        return '3 SHARE & OTHER INVESTMENT'
    elif re.search(r'\bloans?\s*(?:&|and)\s+advances?\b', s_lower):
        return '4 LOANS & ADVANCES'
    elif re.search(r'\bbills?\s+payable\b', s_lower) or s_lower == '4 bills':
        return '4 BILLS PAYABLE'
    elif re.search(r'\bother\s+liabilities\b', s_lower):
        return '5 OTHER LIABILITIES'
    elif (re.search(r'\bbills?\s+purchased?\b', s_lower)
          or s_lower in ['5 bill', '4.2 bill']):
        return '5 BILL PURCHASED'
    elif re.search(r'\bloans\s+against\s+collected\s+bills?\b', s_lower):
        return '6 LOANS AGAINST COLLECTED BILLS'
    elif re.search(r'\bagainst\s+(domestic|foreign)\s+bills?\b', s_lower):
        return '6.a Loans Against Collected Bills - Domestic/Foreign'
    elif re.search(r'\bfixed\s+assets?\b', s_lower):
        return '7 FIXED ASSETS'
    elif re.search(r'\bother\s+assets?\b', s_lower):
        return '8 OTHER ASSETS'
    elif re.search(r'\bexpenses?\s+not\s+written\s+off\b', s_lower):
        return '9 EXPENSES NOT WRITTEN OFF'
    elif re.search(r'\bnon\s+banking(\s+assets?)?\b', s_lower):
        return '10 NON BANKING ASSETS'
    elif re.search(r'\brecon\w*\s+a(?:ccount|/?c)?\b', s_lower):  # tolerates "Reconcillation" misspelling
        return '6 RECONCILIATION ACCOUNT'
    elif re.search(r'\bprofit\s*&\s*loss\b', s_lower):
        return '7 PROFIT & LOSS A/C'
    elif re.search(r'\bcalls?\s+in\s+advance\b', s_lower):
        return '8 CALLS IN ADVANCE'

    # --- Capital Fund sub-items ---
    elif re.search(r'\bpaid[- ]up\s+capital\b', s_lower):
        return '1.a Paid-up Capital'
    elif re.search(r'\bgeneral\s+reserves?\b', s_lower):
        return '1.d General Reserves'
    elif re.search(r'\bretained\s+(earning|profit)\b', s_lower) or s_lower == 'f. retained':
        return '1.f Retained Earnings'
    elif re.search(r'\bothers?\s+reserves?\s+fund\b', s_lower):
        return '1.g Other Reserves Fund'
    elif re.search(r'\bshare\s+premium\b', s_lower) or re.sub(r'^\(?[a-z0-9]{1,2}\)?[.\)]\s*', '', s_lower).strip(' :') == 'share':
        return '1.e Share Premium'
    elif re.search(r'\bproposed\s+bonus\s+share\b', s_lower):
        return '1.c Proposed Bonus Share'

    # --- Borrowings sub-items ---
    # "a. NRB"/"b. NRB" already satisfy \bnrb\b below. The full-sentence
    # "1. In Nepal Rastra Bank" wording (older template, pre-"NRB" abbreviation)
    # is unambiguous and handled directly. Its truncated form "1. In Nepal
    # Rastra" (missing "Bank") is NOT handled directly here -- it collides with
    # a truncated Liquid-Funds "Bank Balance - NRB" row that wraps to the same
    # text, so it's tagged as an ambiguous fragment and resolved by
    # resolve_ambiguous_fragments() based on which section it actually sits
    # under (see _NEPAL_RASTRA_PARENT_OVERRIDES).
    elif s_lower == '1. in nepal rastra bank':
        return '2.a Borrowings - NRB'
    elif s_lower == '1. in nepal rastra':
        return 'AMBIGUOUS_FRAGMENT: NEPAL RASTRA'
    elif re.search(r'\bnrb\b', s_lower) and not re.search(r'\bbond\b', s_lower):
        return '2.a Borrowings - NRB'
    elif re.search(r'\binterbank\s+borrowing\b', s_lower):
        return '2.b Interbank Borrowing'
    elif re.search(r'\bforeign\s+banks?\b', s_lower):
        return '2.c Foreign Banks and Financial Institutions'
    elif re.search(r'\bother\s+financial\s+ins', s_lower):
        return '2.d Other Financial Institutions'
    elif re.search(r'\bbonds?\s+and\s+securities\b', s_lower) or re.search(r'\bnrb\s+bond\b', s_lower):
        return '2.e Bonds and Securities / NRB Bonds'
    elif re.search(r'\bgovt\.?\s*non[- ]fin\.?\s+ins', s_lower):
        return '2.f Borrowings - Govt Non-Financial Institutions'  # best-guess label, verify against source template
    elif re.search(r'\bother\s+non[- ]fin\s+ins', s_lower):
        return '2.g Borrowings - Other Non-Financial Institutions'  # best-guess label, verify against source template

    # --- Deposits sub-items ---
    elif re.search(r'\bsavings?\b', s_lower):
        return '3.b Deposits - Savings'
    elif re.search(r'\b(call\s+deposits?|money\s+at(\s+call)?|call)\b', s_lower):
        # bare "d. Call" / "c. Money at" are truncated PDF wraps of this same
        # sub-item -- safe to catch broadly here since '3 DEPOSITS' top-level
        # already excludes anything containing "call", and "Calls in Advance"
        # (checked earlier) requires "advance" too, so there's no shadowing.
        return '3.d Deposits - Call / Money at Call'
    elif re.search(r'\bnon\s+residents?\b', s_lower):
        return '3.e Deposits - Non Residents'
    elif re.search(r'\bcurrent\b', s_lower):
        return '3.a Deposits - Current'
    elif re.search(r'\bfixed\b', s_lower):  # safe here: "fixed assets" already handled above
        return '3.c Deposits - Fixed'

    # --- Other Liabilities & Provisions sub-items ---
    elif re.search(r'\bsundry\s+creditors\b', s_lower):
        return '5.1 Sundry Creditors'
    elif re.search(r'\bloan\s+loss(\s+provision)?\b', s_lower):
        return '5.2 Loan Loss Provision'
    elif re.search(r'\binterest\s+suspense\b', s_lower):
        return '5.3 Interest Suspense a/c'

    # --- Liquid Funds sub-items ---
    elif re.search(r'\bcash\s+balance\b', s_lower) or re.search(r'\bnepalese\s+notes\b', s_lower):
        return '1.a Cash Balance'
    elif re.search(r'\bcash\s+in\s+transit\b', s_lower):
        return '1.a.1 Cash Balance - Cash in Transit'
    elif re.search(r'\bnepal\s+rastra\s+bank\b', s_lower):
        return '1.b.1 Bank Balance - NRB'
    elif re.search(r'\bclass\s+licensed\b', s_lower) or re.search(r'\ba\s+class\b', s_lower):
        return '1.b.2 Bank Balance - Class A'
    elif re.search(r'\bbank\s+balance\b', s_lower):
        return '1.b Bank Balance'

    # --- Investments ---
    elif re.search(r'\bgovt\.?\s*securities\b', s_lower):
        return '2.a Government Securities'

    # --- Loans & Advances / Bills Purchased sub-items ---
    elif re.search(r'\bstaff\s+loans?\s*/?\s*adv', s_lower):
        return '4.a Loans & Advances - Staff Loans/Advances'
    elif re.search(r'\b(domestic|foreign|import)\s+bills?\b', s_lower):
        return '5 BILL PURCHASED'  # Domestic/Foreign split collapsed; see note below
    elif re.search(r'\bimport\s+bills?\s*&\s*imports?\b', s_lower):
        return '5 BILL PURCHASED'

    # --- Other Assets sub-items ---
    elif re.search(r'\baccrued\s+interest\b', s_lower):
        return '8.a Other Assets - Accrued Interest'
    elif re.search(r'\bsundry\s+debtors\b', s_lower):
        return '8.b Other Assets - Sundry Debtors'

    # --- Off-balance-sheet / marginals ---
    elif re.search(r'\binterbank\s+lending\b', s_lower):
        return 'INTERBANK LENDING (Marginal r_i)'

    # Bare, context-dependent words that mean different things depending on
    # which section they physically sit under in the source PDF (e.g.
    # "Domestic" under Deposits>Current vs. under Bills Purchased). Regex on
    # the text alone can't resolve these — tagged for the stateful pass in
    # resolve_ambiguous_fragments() instead of silently returning s.upper().
    elif (stripped_fragment := re.sub(r'^\(?[a-z0-9]{1,2}\)?[.\)]\s*', '', s_lower).strip(' :')) in [
        'domestic', 'foreign', 'foreign currency', 'domestic currency',
        'private', 'private sector', 'financial', 'financial institutions',
        'government', 'government enterprises', 'government organizations',
        'others', 'sundry', 'non',
    ]:
        # Same concept, different wording across fiscal-year templates — merge
        # the qualifier variant into the bare form so they land in one column
        # instead of splitting (confirmed via null-count comparison: e.g.
        # "Domestic" and "Domestic Currency" under the same parent section
        # partition the same 512 rows with no overlap).
        normalized = _FRAGMENT_ALIASES.get(stripped_fragment, stripped_fragment)
        return f'AMBIGUOUS_FRAGMENT: {normalized.upper()}'

    return s.upper()


def build_paper1_features(pivot: pd.DataFrame) -> pd.DataFrame:
    """Compute Size (X1) and the 4 ratio covariates (X2-X5) from a pivoted
    balance-sheet feature table. Call this BEFORE dropping legacy_cols, so
    Size reflects true total assets regardless of which top-level columns
    later get excluded as standalone covariates.
    """
    out = pivot[['year_bs', 'month_bs', 'bfi_name']].copy()
    asset_cols = [c for c in ASSET_TOP_LEVEL if c in pivot.columns]
    out['size'] = pivot[asset_cols].sum(axis=1, min_count=1)
    out['capital_ratio'] = pivot['1 CAPITAL FUND'] / out['size']
    out['liquidity_ratio'] = pivot['1 LIQUID FUNDS'] / out['size']
    out['loan_to_deposit'] = pivot['4 LOANS & ADVANCES'] / pivot['3 DEPOSITS']
    borrow_sub_cols = [c for c in BORROWINGS_SUBITEMS if c in pivot.columns]
    borrowings_total = pivot['2 BORROWINGS'].fillna(pivot[borrow_sub_cols].sum(axis=1, min_count=1))
    out['nrb_dependence'] = pivot['2.a Borrowings - NRB'] / borrowings_total
    out['r_i'] = pivot['INTERBANK LENDING (Marginal r_i)']
    out['c_i'] = pivot['2.b Interbank Borrowing']
    out['K_i'] = pivot['1 CAPITAL FUND']
    print("=="*50)
    print(f"RAW COLUMNS: {asset_cols},'1 CAPITAL FUND', \
          '1 LIQUID FUNDS', '4 LOANS & ADVANCES', '3 DEPOSITS', {borrow_sub_cols},  \
            'INTERBANK LENDING (Marginal r_i)','2 BORROWINGS', \
            '2.a Borrowings - NRB', '1 CAPITAL FUND'")
    print("=="*50)

    return out


def resolve_ambiguous_fragments(df: pd.DataFrame, line_item_col: str = 'line_item') -> pd.DataFrame:
    """Rewrite 'AMBIGUOUS_FRAGMENT: X' rows as 'PARENT > X' using the nearest
    preceding top-level row within each (source_file, statement, bfi_name)
    group. Assumes df is already in original PDF row order (e.g. sorted by
    page/original index) and line_item_col already holds canonicalize_line_item_regex output.
    """
    group_cols = [c for c in ['source_file', 'statement', 'bfi_name'] if c in df.columns]
    groups = df.groupby(group_cols).groups if group_cols else {None: df.index}

    resolved = df[line_item_col].copy()
    for _, idx in groups.items():
        current_parent = 'UNKNOWN SECTION'
        for i in idx:
            value = df.at[i, line_item_col]
            if value in _TOP_LEVEL_LABELS:
                current_parent = value
            elif isinstance(value, str) and value.startswith('AMBIGUOUS_FRAGMENT: '):
                fragment = value[len('AMBIGUOUS_FRAGMENT: '):]
                if fragment == 'SUNDRY' and current_parent in _SUNDRY_PARENT_OVERRIDES:
                    resolved.at[i] = _SUNDRY_PARENT_OVERRIDES[current_parent]
                elif fragment == 'NEPAL RASTRA' and current_parent in _NEPAL_RASTRA_PARENT_OVERRIDES:
                    resolved.at[i] = _NEPAL_RASTRA_PARENT_OVERRIDES[current_parent]
                else:
                    resolved.at[i] = f'{current_parent} > {fragment}'

    df = df.copy()
    df[line_item_col] = resolved
    return df


def find_candidate_duplicate_columns(
    wide_df: pd.DataFrame,
    meta_cols=('year_bs', 'month_bs', 'bfi_name'),
    max_overlap: float = 0.05,
    min_similarity: float = 0.5,
) -> pd.DataFrame:
    """Scan a pivoted (wide) feature table for column pairs that are likely the
    same concept under two different era wordings: they rarely have data at
    the same time (mutually exclusive across the panel) AND their text is
    similar. Returns candidates for manual review, not auto-merged — a
    mutually-exclusive, similar-looking pair can still be two genuinely
    distinct categories (e.g. 'Government Enterprises' vs 'Government
    Organizations') that just happen to both be rare.
    """
    import itertools
    from difflib import SequenceMatcher

    feature_cols = [c for c in wide_df.columns if c not in meta_cols]
    rows = []
    for col_a, col_b in itertools.combinations(feature_cols, 2):
        mask_a = wide_df[col_a].notna()
        mask_b = wide_df[col_b].notna()
        n_a, n_b = mask_a.sum(), mask_b.sum()
        if n_a == 0 or n_b == 0:
            continue
        overlap_frac = (mask_a & mask_b).sum() / min(n_a, n_b)
        if overlap_frac > max_overlap:
            continue
        frag_a = col_a.split('>')[-1].strip()
        frag_b = col_b.split('>')[-1].strip()
        similarity = SequenceMatcher(None, frag_a, frag_b).ratio()
        if similarity >= min_similarity:
            parent_a = col_a.split('>')[0].strip() if '>' in col_a else col_a
            parent_b = col_b.split('>')[0].strip() if '>' in col_b else col_b
            rows.append({
                'col_a': col_a, 'col_b': col_b,
                'same_parent': parent_a == parent_b,
                'non_null_a': int(n_a), 'non_null_b': int(n_b),
                'overlap_frac': round(overlap_frac, 3),
                'text_similarity': round(similarity, 3),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # same_parent candidates are the real merge signal (same bare word under a
    # different parent, like row 1 above, usually means a stale current_parent
    # in resolve_ambiguous_fragments rather than a true duplicate — investigate
    # those separately instead of merging them).
    return result.sort_values(['same_parent', 'text_similarity'], ascending=[False, False]).reset_index(drop=True)


def find_missing_top_level_labels(df: pd.DataFrame, line_item_col: str = 'line_item') -> list:
    """Catch the exact bug class that produced '4 BILLS PAYABLE > OTHERS': a
    header-shaped canonical value (leading digit, no sub-level '.' like '1.a'
    or '5.1') that resolve_ambiguous_fragments doesn't recognize as a parent
    because it's missing from _TOP_LEVEL_LABELS. Any such value silently stalls
    current_parent for every row after it until the next recognized header, so
    this should be re-run any time canonicalize_line_item_regex gains a new
    top-level branch or _TOP_LEVEL_LABELS is edited.
    """
    header_shape = re.compile(r'^\d+\s[A-Z]')
    seen = df[line_item_col].dropna().unique()
    return sorted(
        v for v in seen
        if isinstance(v, str) and header_shape.match(v) and v not in _TOP_LEVEL_LABELS
    )


def coalesce_columns(wide_df: pd.DataFrame, keep: str, drop: list) -> pd.DataFrame:
    """Merge confirmed duplicate columns into `keep` via combine_first (first
    non-null wins), then drop the variant columns. Only call this on pairs
    you've reviewed from find_candidate_duplicate_columns, not the full list.
    """
    wide_df = wide_df.copy()
    for col in drop:
        wide_df[keep] = wide_df[keep].combine_first(wide_df[col])
    return wide_df.drop(columns=drop)
