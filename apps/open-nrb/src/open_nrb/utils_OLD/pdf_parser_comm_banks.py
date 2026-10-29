#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pdfplumber>=0.11.0",
#     "pandas>=2.0.0",
# ]
# ///

import argparse
import json
from pathlib import Path
import pdfplumber

def extract_commercial_banks_dict(pdf_path: Path) -> dict:
    """Extracts commercial banks into a {"abbreviation": "fullname"} dictionary."""
    bank_dict = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        target_pages = [
            i for i, page in enumerate(pdf.pages) 
            if "Annex-I" in (page.extract_text() or "") or "Name list of Bank" in (page.extract_text() or "")
        ]
        
        if not target_pages:
            target_pages = list(range(max(0, len(pdf.pages) - 15), len(pdf.pages)))

        for p_idx in target_pages:
            page = pdf.pages[p_idx]
            tables = page.extract_tables()
            is_commercial_section = False
            
            for table in tables:
                for row in table:
                    cleaned = [str(cell).strip().replace('\n', ' ') if cell else "" for cell in row]
                    joined = " ".join(cleaned)
                    
                    if "Commercial Banks" in joined or "A. Commercial" in joined:
                        is_commercial_section = True
                        continue
                    elif "Development Banks" in joined or "B. Development" in joined:
                        is_commercial_section = False
                        continue
                        
                    if is_commercial_section:
                        # Left side: [SN, Full Name, Short Name] (indices 1, 2)
                        if len(cleaned) >= 3 and cleaned[1] and not cleaned[1].isdigit():
                            full_name = cleaned[1]
                            short_name = cleaned[2] if len(cleaned) > 2 else ""
                            if short_name and not short_name.isdigit():
                                bank_dict[short_name] = full_name
                                
                        # Right side: [SN, Full Name, Short Name] (indices 4, 5)
                        if len(cleaned) >= 6 and cleaned[4] and not cleaned[4].isdigit():
                            full_name = cleaned[4]
                            short_name = cleaned[5] if len(cleaned) > 5 else ""
                            if short_name and not short_name.isdigit():
                                bank_dict[short_name] = full_name
                                
    return bank_dict

def run_sanity_check_and_save(raw_dir: Path, output_path: Path, max_sanity_files: int = 300):
    pdf_files = sorted(list(raw_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {raw_dir}")
        return

    print(f"Running sanity check on up to {max_sanity_files} sample files...")
    master_mapping = {}
    
    # Sanity check validation keys that must exist in a valid extraction
    mandatory_keys = {"NBL", "RBB", "NABIL"}

    for pdf_path in pdf_files[:max_sanity_files]:
        print(f" -> Inspecting: {pdf_path.name}")
        mapping = extract_commercial_banks_dict(pdf_path)
        
        # Sanity check assertions
        missing = mandatory_keys - mapping.keys()
        if missing:
            print(f"    [WARNING] Sanity check failed for {pdf_path.name}. Missing keys: {missing}")
        else:
            print(f"    [PASSED] Successfully extracted {len(mapping)} commercial banks with valid core keys.")
            
        # Merge into master dictionary (latest entries or overwrite)
        master_mapping.update(mapping)

    # Save final aggregated mapping file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master_mapping, f, indent=4, ensure_ascii=False)
    
    print(f"\nSaved master commercial bank mapping to: {output_path}")
    print(f"Total unique commercial banks mapped: {len(master_mapping)}")

def main():
    parser = argparse.ArgumentParser(description="Extract and validate Commercial Bank Mappings from NRB PDFs")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing monthly statistical PDF files")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    
    args = parser.parse_args()

    default_dir = Path(__file__).resolve().parents[3] / "src" / "open_nrb" / "data" / "raw_pdfs" / "monthly_statistical_bfis"
    output = Path(__file__).resolve().parents[3] / "src" / "open_nrb" / "data" / "processed" / "commercial_banks_mapping.json"
  
    target_dir = Path(args.dir) if args.dir else default_dir
    out_path = Path(args.output) if args.output else output

    run_sanity_check_and_save(target_dir, out_path)

if __name__ == "__main__":
    main()