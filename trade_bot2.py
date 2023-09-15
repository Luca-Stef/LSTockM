import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL to scrape
url = "https://finviz.com/insidertrading.ashx"

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the insider trading data
    table = soup.find('table', {'class': 'body-table'})

    # Extract data from the table and store it in a pandas DataFrame
    df = pd.read_html(str(table))[0]

    # Print the DataFrame
    print(df)

else:
    print("Failed to retrieve the web page.")