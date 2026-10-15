#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests",
#     "beautifulsoup4",
# ]
# ///

import time
import random
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote
from pathlib import Path

def robust_get(url, headers, max_retries=3, timeout=30):
    """Helper function to fetch URLs with retries and higher timeout."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if response.status_code == 200:
                return response
            elif response.status_code == 404:
                return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == max_retries:
                return None
            time.sleep(3)
        except Exception:
            return None
    return None

def get_safe_filename(pdf_url: str, max_byte_length: int = 120) -> str:
    """
    Sanitizes and safely truncates filenames to prevent Linux [Errno 36] File name too long.
    Uses a URL hash fallback if the name is too long or contains heavy unicode encoding.
    """
    raw_name = pdf_url.split('/')[-1].split('?')[0]
    decoded_name = unquote(raw_name)
    
    # Keep only safe characters (alphanumeric, underscores, hyphens, dots)
    clean_name = "".join(c for c in decoded_name if c.isalnum() or c in ('_', '-', '.'))
    
    # Check byte length (Linux limit is 255 bytes per filename)
    name_bytes = clean_name.encode('utf-8')
    if len(name_bytes) > max_byte_length or not clean_name or not clean_name.lower().endswith('.pdf'):
        url_hash = hashlib.md5(pdf_url.encode('utf-8')).hexdigest()[:10]
        # Keep a short clean prefix if available
        prefix = clean_name[:30].strip('-_')
        clean_name = f"{prefix}_{url_hash}.pdf" if prefix else f"nrb_doc_{url_hash}.pdf"
        
    return clean_name

def download_category_pdfs(category_name, config, base_output_dir):
    """
    Scrapes and downloads PDFs for a specific NRB report category into its own directory
    with resume protection, partial-file handling, and length safety.
    """
    output_dir = base_output_dir / category_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"\n==================================================")
    print(f" Processing Category: {category_name.upper()}")
    print(f" Target Directory: {output_dir.resolve()}")
    print(f"==================================================")

    total_downloaded = 0
    total_skipped = 0

    max_pages = config.get("max_pages", 5)
    slug_filter = config.get("slug_filter", "")
    valid_paths = config.get("valid_paths", ["/red/", "/bfr/", "/bsd/"])

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            cat_url = config["base_url"]
        else:
            cat_url = config["page_template"].format(page_num)

        print(f"\nScanning Page {page_num}: {cat_url}")
        
        cat_res = robust_get(cat_url, headers)
        if not cat_res:
            print(f"  -> Page {page_num} could not be reached or does not exist. Ending pagination for this category.")
            break
            
        soup = BeautifulSoup(cat_res.text, 'html.parser')
        detail_urls = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            if 'category' in href or 'page' in href or href == "https://www.nrb.org.np/":
                continue
            
            is_valid_path = any(path in href for path in valid_paths)
            is_valid_slug = slug_filter in href if slug_filter else True
            
            if (is_valid_path or is_valid_slug) and href not in detail_urls:
                detail_urls.append(href)

        print(f"  -> Found {len(detail_urls)} report items on this page.")
        if len(detail_urls) == 0:
            break

        for idx, detail_url in enumerate(detail_urls, start=1):
            detail_res = robust_get(detail_url, headers)
            if not detail_res:
                print(f"     [{idx}/{len(detail_urls)}] Timeout reading detail URL: {detail_url}")
                continue
                
            pdf_url = None
            
            if detail_res.url.lower().endswith('.pdf') or 'application/pdf' in detail_res.headers.get('content-type', '').lower():
                pdf_url = detail_res.url
            else:
                detail_soup = BeautifulSoup(detail_res.text, 'html.parser')
                for a_tag in detail_soup.find_all('a', href=True):
                    href = a_tag['href']
                    if '.pdf' in href.lower():
                        full_url = urljoin(detail_url, href)
                        if '.pdf' in full_url.lower():
                            pdf_url = full_url
                            break
            
            if not pdf_url:
                print(f"     [{idx}/{len(detail_urls)}] No PDF link found in {detail_url}")
                time.sleep(random.uniform(0.3, 0.7))
                continue
                
            # Safely generate filename avoiding length and unicode crashes
            filename = get_safe_filename(pdf_url)
            file_path = output_dir / filename
            part_path = output_dir / f"{filename}.part"
            
            if file_path.exists() and file_path.stat().st_size > 1024:
                print(f"     [{idx}/{len(detail_urls)}] Exists (valid), skipping: {filename}")
                total_skipped += 1
            else:
                if part_path.exists():
                    part_path.unlink()

                print(f"     [{idx}/{len(detail_urls)}] Downloading: {filename}...")
                try:
                    success = False
                    if detail_res.url == pdf_url:
                        with open(part_path, 'wb') as f:
                            f.write(detail_res.content)
                        success = True
                    else:
                        pdf_res = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
                        if pdf_res.status_code == 200:
                            with open(part_path, 'wb') as f:
                                for chunk in pdf_res.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            success = True
                        else:
                            print(f"       -> Failed download. Status code: {pdf_res.status_code}")
                    
                    if success and part_path.exists() and part_path.stat().st_size > 0:
                        part_path.rename(file_path)
                        total_downloaded += 1
                    else:
                        print(f"       -> Download incomplete or empty file.")
                        if part_path.exists():
                            part_path.unlink()

                except Exception as e:
                    print(f"       -> Error downloading file: {e}")
                    if part_path.exists():
                        part_path.unlink()
                
                time.sleep(random.uniform(1.0, 1.5))
                
        time.sleep(random.uniform(1.0, 2.0))

    print(f"\nFinished category [{category_name}] -> Downloaded: {total_downloaded} | Skipped: {total_skipped}")

def main():
    current_file = Path(__file__).resolve()
    base_output_dir = current_file.parent.parent / "data" / "raw_pdfs"

    categories = {
        "study_reports": {
            "base_url": "https://www.nrb.org.np/category/study-reports/?department=red",
            "page_template": "https://www.nrb.org.np/category/study-reports/page/{}/?department=red",
            "max_pages": 10,
            "slug_filter": "study-reports",
            "valid_paths": ["/red/"]
        },
        "financial_stability_report": {
            "base_url": "https://www.nrb.org.np/category/financial-stability-report/",
            "page_template": "https://www.nrb.org.np/category/financial-stability-report/page/{}/",
            "max_pages": 5,
            "slug_filter": "financial-stability-report",
            "valid_paths": ["/bfr/", "/red/"]
        },
        "annual_reports": {
            "base_url": "https://www.nrb.org.np/category/annual-reports/?department=red",
            "page_template": "https://www.nrb.org.np/category/annual-reports/page/{}/?department=red",
            "max_pages": 10,
            "slug_filter": "annual-reports",
            "valid_paths": ["/red/"]
        },
        "key_financial_indicators": {
            "base_url": "https://www.nrb.org.np/category/key-financial-indicators/?department=bsd",
            "page_template": "https://www.nrb.org.np/category/key-financial-indicators/page/{}/?department=bsd",
            "max_pages": 5,
            "slug_filter": "key-financial-indicators",
            "valid_paths": ["/bsd/"]
        }
    }

    for cat_name, config in categories.items():
        download_category_pdfs(cat_name, config, base_output_dir)

    print("\n==================================================")
    print(" All category pipelines completed successfully!")
    print(f" Files stored under: {base_output_dir.resolve()}")
    print("==================================================")

if __name__ == "__main__":
    main()