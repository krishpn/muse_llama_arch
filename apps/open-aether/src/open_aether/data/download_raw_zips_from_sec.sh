# run on the box, next to wdrs_data
mkdir -p sec_insider && cd sec_insider
UA="<First_Name Last_Name> (academic research) <academic email>"   # SEC REQUIRES a real UA

# 1. grab the listing page and extract all zip hrefs (absolute + site-relative)
curl -s -A "$UA" \
https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets \
-o index.html
grep -oiE 'href="[^"]+\.zip"' index.html \
| sed -E 's/.*href="//I; s/"$//' \
| sed -E 's#^/#https://www.sec.gov/#' \
| sort -u > ziplist.txt
wc -l ziplist.txt && head ziplist.txt      # sanity-check the URLs look right

# 2. download throttled (well under SEC's 10 req/sec), resumable
mkdir -p zips
while read -r url; do
fn=$(basename "$url")
[ -s "zips/$fn" ] && { echo "skip $fn"; continue; }   
echo "fetching $fn"
curl -s -A "$UA" "$url" -o "zips/$fn"
sleep 0.5
done < ziplist.txt

# 3. unzip each into its own quarter folder
for z in zips/*.zip; do d="${z%.zip}"; mkdir -p "extracted/$(basename "$d")"; unzip -oq "$z" -d "extracted/$(basename "$d")"; done



cd ~/Documents/sec_insider/extracted     # adjust to where these live

# schema of the key tables, early era vs post-amendment era
for q in 2006q1_form345 2024q1_form345; do
echo "########## $q ##########"
echo "--- SUBMISSION ---";     head -1 "$q/SUBMISSION.tsv"     | tr '\t' '\n' | nl
echo "--- REPORTINGOWNER ---"; head -1 "$q/REPORTINGOWNER.tsv" | tr '\t' '\n' | nl
echo "--- NONDERIV_TRANS ---"; head -1 "$q/NONDERIV_TRANS.tsv" | tr '\t' '\n' | nl
done

# locate the 10b5-1 column anywhere in a post-amendment quarter
echo "########## 10b5-1 search (2024q1) ##########"
for f in 2024q1_form345/*.tsv; do
    hit=$(head -1 "$f" | tr '\t' '\n' | grep -in '10b5\|rule\|plan')
    [ -n "$hit" ] && echo "$f: $hit"
done