from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import json
class TablePreprocessor:
    def __init__(self):
        pass

    def parse_html_table(self, html_string: str) -> str:
        """Convert HTML table to plain text: rows tab-separated."""
        soup = BeautifulSoup(html_string, 'html.parser')
        tables = soup.find_all('table')
        table_texts = []
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all(['td', 'th'])
                row_text = [col.get_text(strip=True) for col in cols]
                table_texts.append('\t'.join(row_text))
        return '\n'.join(table_texts)

    def html_table_to_markdown(self, html_string: str) -> str:
        """Convert HTML table to Markdown table."""
        soup = BeautifulSoup(html_string, 'html.parser')
        tables = soup.find_all('table')
        md_tables = []
        for table in tables:
            rows = table.find_all('tr')
            md = []
            for i, row in enumerate(rows):
                cols = [col.get_text(strip=True) for col in row.find_all(['td', 'th'])]
                line = '| ' + ' | '.join(cols) + ' |'
                md.append(line)
                if i == 0:
                    md.append('|' + '|'.join(['---'] * len(cols)) + '|')
            md_tables.append('\n'.join(md))
        return '\n\n'.join(md_tables)

    def extract_table_tags_as_text(self, text: str) -> str:
        """Find <<TABLE:.../TABLE>> tags and replace with plain text tables."""
        pattern = r"<<TABLE:(.*?)\/TABLE>>"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for match in matches:
            parsed = self.parse_html_table(match)
            text = text.replace(f"<<TABLE:{match}/TABLE>>", parsed)
        return text

    def extract_table_tags_as_markdown(self, text: str) -> str:
        """Find <<TABLE:.../TABLE>> tags and replace with Markdown tables."""
        pattern = r"<<TABLE:(.*?)\/TABLE>>"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for match in matches:
            parsed = self.html_table_to_markdown(match)
            text = text.replace(f"<<TABLE:{match}/TABLE>>", parsed)
        return text

# if __name__ == "__main__":
#     path = os.path.join("data", "VLSP2025", "law_db", "vlsp2025_law.json")
#     with open(path, "r", encoding="utf-8") as f:
#         law_data = json.load(f)
#     tp = TablePreprocessor()
#     all_texts= []
#     for law in law_data:
#         law_id = law["id"].replace(" ", "_").replace("/", "_")
#         for article in law["articles"]:
#             text = article.get("text", "")
#             article_id = article["id"]
#             test = tp.extract_table_tags_as_markdown(text)
#             print(test)
#             all_texts.append(f"{law_id}_{article_id}:\n{test}")
#     output_path = 'data/all_laws_clean.txt'
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write('\n'.join(all_texts))


