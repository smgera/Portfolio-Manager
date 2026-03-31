import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time  # Import time to use sleep for waiting between page loads

# Base URL of the page to scrape
base_url = 'http://stockanalysis.com/stocks/'

# Set up Selenium WebDriver
options = Options()
options.headless = True  # Run in headless mode
options.binary_location = 'C:/Program Files/Google/Chrome/Application/chrome.exe'  # Update with the correct path to your Chrome executable
service = Service('C:/path/to/chromedriver')  # Update with the correct path to your ChromeDriver
driver = webdriver.Chrome(service=service, options=options)
driver.get(base_url)

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
max_iterations = 15
iterations = 0

while iterations < max_iterations:
    # Get the page source and parse it with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    headers, rows = extract_table_data(soup)
    if not rows:
        print(f"No table found on page {page_number}")
        break
    
    # Append the data to the DataFrame
    page_data = pd.DataFrame(rows, columns=headers)
    all_data = pd.concat([all_data, page_data], ignore_index=True)
    
    # Check if there is a next page
    try:
        next_button = driver.find_element(By.XPATH, "//button[contains(@class, 'controls-btn') and contains(., 'Next')]")
        if 'disabled' in next_button.get_attribute('class'):
            print(f"'Next' button is disabled on page {page_number}")
            break
        next_button.click()
        time.sleep(1)  # Wait for the next page to load
    except Exception as e:
        print(f"No 'Next' button found on page {page_number}: {e}")
        break
    
    page_number += 1
    iterations += 1

# Save the data to a CSV file
all_data.to_csv('stock_data.tsv', sep='\t', index=False)
print(all_data)

# Close the WebDriver
driver.quit()