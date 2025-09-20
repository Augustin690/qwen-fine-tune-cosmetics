"""Script to analyse scrape.log to spot faulty product URLs."""

import re
import pandas as pd
from collections import defaultdict

def analyze_log(log_file_path: str, output_csv_path: str) -> None:
    # Dictionary to store error counts and problematic URLs
    error_summary = defaultdict(lambda: {"count": 0, "urls": set()})

    # Regex to extract log level, message, and URL
    log_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\w+) - (.*)")

    # Read the log file
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            match = log_pattern.match(line)
            if not match:
                continue

            timestamp, level, message = match.groups()

            # Focus on ERROR level logs
            if level == "ERROR":
                # Extract the URL from the error message (if present)
                url_match = re.search(r"https?://[^\s]+", message)
                if url_match:
                    url = url_match.group(0)
                    # Update error summary
                    error_summary[message]["count"] += 1
                    error_summary[message]["urls"].add(url)

    # Convert the error summary to a DataFrame
    error_data = []
    for error_message, details in error_summary.items():
        error_data.append({
            "error_message": error_message,
            "count": details["count"],
            "urls": ", ".join(details["urls"])
        })

    df = pd.DataFrame(error_data)

    # Save the results to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Error analysis saved to {output_csv_path}")

# Example usage
analyze_log(
    log_file_path="../log/scraper.log",
    output_csv_path="../log/error_summary.csv"
)