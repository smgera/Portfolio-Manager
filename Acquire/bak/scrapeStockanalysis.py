import pandas as pd
import requests 
from bs4 import BeautifulSoup

# Base URL of the page to scrape
base_url = 'http://stockanalysis.com/stocks/'

# Function to get the soup object for a given URL
def get_soup(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BeautifulSoup(response.content, 'html.parser')
    else:
        return None

# Function to extract table data from a soup object
def extract_table_data(soup):
    table = soup.find('table')
    headers = [header.text for header in table.find_all('th')]
    rows = []
    for row in table.find_all('tr')[1:]:  # Skip the header row
        rows.append([cell.text for cell in row.find_all('td')])
    return headers, rows

# Initialize an empty DataFrame to store all the data
all_data = pd.DataFrame()

# Start with the first page
page_number = 1
while True:
    url = f"{base_url}?page={page_number}"
    print(url)
    soup = get_soup(url)
    if soup is None:
        print(f"Could not get data from {url}")
        break
    
    headers, rows = extract_table_data(soup)
    if not rows:
        print(f"No table found on page {page_number}")
        break
    
    print(headers)
    # Append the data to the DataFrame
    page_data = pd.DataFrame(rows, columns=headers)
    all_data = pd.concat([all_data, page_data], ignore_index=True)
    
    # Check if there is a next page
    next_page = soup.find('a', {'aria-label': 'Next'})
    if not next_page or 'disabled' in next_page.get('class', []):
        print('no next page')
        break
    
    page_number += 1

# Save the data to a CSV file
all_data.to_csv('stock_data.tsv', sep='\t', index=False)