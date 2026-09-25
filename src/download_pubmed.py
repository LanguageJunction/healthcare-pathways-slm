import requests
import time
from pathlib import Path
import xml.etree.ElementTree as ET


# =========================================================
# Configuration
# =========================================================

OUTPUT_FILE = "data/medical.txt"

# Number of PubMed abstracts to download
MAX_ARTICLES = 5_000

# Articles per API request
BATCH_SIZE = 200

# Medical/biomedical search query
QUERY = """
(
    medicine
    OR healthcare
    OR medical device
    OR disease
    OR diagnosis
    OR treatment
    OR clinical
    OR pharmacology
    OR surgery
    OR oncology
    OR cardiology
    OR diabetes
    OR hypertension
)
AND hasabstract[text]
"""

EMAIL = "your_email@example.com"


BASE_URL = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/"
)


# =========================================================
# Search PubMed
# =========================================================

def search_pubmed(query, max_articles):

    url = BASE_URL + "esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_articles,
        "retmode": "json",
        "sort": "pub_date",
        "email": EMAIL
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    ids = data["esearchresult"]["idlist"]

    print(
        f"Found {len(ids)} PubMed articles"
    )

    return ids


# =========================================================
# Fetch abstracts
# =========================================================

def fetch_abstracts(pmids):

    url = BASE_URL + "efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": EMAIL
    }

    response = requests.get(
        url,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    return response.text


# =========================================================
# Extract title + abstract
# =========================================================

def extract_text(xml_text):

    root = ET.fromstring(xml_text)

    documents = []

    for article in root.findall(".//PubmedArticle"):

        title_element = article.find(
            ".//ArticleTitle"
        )

        title = ""

        if title_element is not None:

            title = "".join(
                title_element.itertext()
            )

        abstract_parts = []

        for abstract in article.findall(
            ".//Abstract/AbstractText"
        ):

            text = "".join(
                abstract.itertext()
            )

            if text:
                abstract_parts.append(
                    text
                )

        abstract = " ".join(
            abstract_parts
        )

        if title or abstract:

            document = (
                title.strip()
                + "\n"
                + abstract.strip()
            )

            documents.append(
                document
            )

    return documents


# =========================================================
# Main
# =========================================================

def main():

    output_path = Path(
        OUTPUT_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------
    # Search
    # ---------------------------------------------

    pmids = search_pubmed(
        QUERY,
        MAX_ARTICLES
    )

    total = len(pmids)

    print(
        f"Downloading {total} articles..."
    )

    # ---------------------------------------------
    # Download
    # ---------------------------------------------

    article_count = 0

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as output:

        for start in range(
            0,
            total,
            BATCH_SIZE
        ):

            batch = pmids[
                start:start + BATCH_SIZE
            ]

            print(
                f"Downloading "
                f"{start + 1}-{start + len(batch)} "
                f"/ {total}"
            )

            try:

                xml = fetch_abstracts(
                    batch
                )

                documents = extract_text(
                    xml
                )

                for document in documents:

                    output.write(
                        document
                        + "\n\n"
                    )

                    article_count += 1

            except Exception as e:

                print(
                    "ERROR:",
                    e
                )

            # Be polite to NCBI
            time.sleep(0.34)

    print()
    print(
        f"Saved {article_count} articles"
    )

    print(
        f"File: {output_path}"
    )


if __name__ == "__main__":
    main()