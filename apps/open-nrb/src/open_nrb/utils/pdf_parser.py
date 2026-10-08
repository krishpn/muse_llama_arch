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
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path

def robust_get(url, headers, max_retries=3, timeout=30):
    """Helper function to fetch URLs with retries and higher timeout."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if response.status_code == 200:
                return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == max_retries:
                return None
            time.sleep(3)
        except Exception:
            return None
    return None

def download_nrb_monthly_pdfs(output_dir: Path = None, max_pages: int = 19):
    """
    Sequentially processes category pages, handles direct PDF redirects, 
    extracts fallback links, and downloads missing PDFs.
    """
    if output_dir is None:
        current_file = Path(__file__).resolve()
        output_dir = current_file.parent.parent / "data" / "raw_pdfs"
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_url_template = "https://www.nrb.org.np/category/monthly-statistics/page/{}/?department=bfr"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"Target data directory: {output_dir.resolve()}")
    print("--- Starting Sequential Download Pipeline ---\n")

    total_downloaded = 0
    total_skipped = 0

    for page_num in range(1, max_pages + 1):
        cat_url = category_url_template.format(page_num)
        print(f"=== Scanning Category Page {page_num} of {max_pages} ===")
        
        cat_res = robust_get(cat_url, headers)
        if not cat_res:
            print(f"  -> Failed to load category page {page_num}. Skipping.")
            continue
            
        soup = BeautifulSoup(cat_res.text, 'html.parser')
        detail_urls = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if '/bfr/' in href and href != 'https://www.nrb.org.np/bfr/':
                if href not in detail_urls:
                    detail_urls.append(href)
                    
        print(f"  -> Found {len(detail_urls)} reports on this page.")

        for idx, detail_url in enumerate(detail_urls, start=1):
            detail_res = robust_get(detail_url, headers)
            if not detail_res:
                print(f"     [{idx}/{len(detail_urls)}] Timeout reading: {detail_url}")
                continue
                
            pdf_url = None
            
            # CASE 1: Did the request URL itself redirect directly to a PDF?
            if detail_res.url.lower().endswith('.pdf') or 'application/pdf' in detail_res.headers.get('content-type', '').lower():
                pdf_url = detail_res.url
            else:
                # CASE 2: Parse HTML page to find the PDF link
                detail_soup = BeautifulSoup(detail_res.text, 'html.parser')
                for a_tag in detail_soup.find_all('a', href=True):
                    href = a_tag['href']
                    if '.pdf' in href.lower() or 'pdf' in a_tag.get('class', []):
                        full_url = urljoin(detail_url, href)
                        if '.pdf' in full_url.lower():
                            pdf_url = full_url
                            break
            
            if not pdf_url:
                print(f"     [{idx}/{len(detail_urls)}] No PDF link found in {detail_url}")
                time.sleep(random.uniform(0.5, 1.0))
                continue
                
            # Check local storage & download
            filename = pdf_url.split('/')[-1].split('?')[0] # Clean query params if any
            file_path = output_dir / filename
            
            if file_path.exists():
                print(f"     [{idx}/{len(detail_urls)}] Exists, skipping: {filename}")
                total_skipped += 1
            else:
                print(f"     [{idx}/{len(detail_urls)}] Downloading: {filename}...")
                try:
                    # If we already have the raw content from a direct redirect, write it directly
                    if detail_res.url == pdf_url:
                        with open(file_path, 'wb') as f:
                            f.write(detail_res.content)
                        total_downloaded += 1
                    else:
                        pdf_res = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
                        if pdf_res.status_code == 200:
                            with open(file_path, 'wb') as f:
                                for chunk in pdf_res.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            total_downloaded += 1
                        else:
                            print(f"       -> Failed download. Status code: {pdf_res.status_code}")
                except Exception as e:
                    print(f"       -> Error downloading file: {e}")
                
                time.sleep(random.uniform(1.0, 2.0))
                
        time.sleep(random.uniform(1.5, 2.5))

    print(f"\nPipeline finished!")
    print(f" - Newly downloaded: {total_downloaded}")
    print(f" - Skipped (already existed): {total_skipped}")

if __name__ == "__main__":
    download_nrb_monthly_pdfs()