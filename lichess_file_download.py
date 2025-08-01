import requests
from datetime import datetime, date
from dateutil.relativedelta import relativedelta


def find_latest_available_database_url(max_months_to_check=6):
    """
    Finds the URL for the most recent available Lichess database,
    checking backwards from last month.
    """
    # Start with the first day of the current month
    current_date = date.today().replace(day=1)

    # Loop backwards through the months
    for i in range(max_months_to_check):
        # Go back i months from the current month (i=0 is last month)
        target_month_date = current_date - relativedelta(months=i + 1)
        year = target_month_date.year
        month = target_month_date.month

        url_date = f"{year}-{month:02d}"
        file_url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{url_date}.pgn.zst"

        print(f"Checking for {target_month_date.strftime('%B %Y')} data at: {file_url}")

        try:
            response = requests.head(file_url)
            # If the file doesn't exist, Lichess returns a 404, which raises an error here
            response.raise_for_status()

            # If we get here, the file exists!
            print(f"✅ Success! Found latest available file.")
            return file_url

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print("-> Not found. Trying previous month.")
                continue  # Go to the next iteration of the loop
            else:
                # For other errors (like 500 server error), we should stop
                print(f"An unexpected HTTP error occurred: {e}")
                return None

    print("❌ Failed. No database file found in the last 6 months.")
    return None


# --- Running the Test ---
# With the current date as August 1, 2025, this will find the June 2025 file.
latest_url = find_latest_available_database_url()

if latest_url:
    print(f"\nLatest URL to use for download: {latest_url}")
