# Importing files
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
import os
from bs4 import BeautifulSoup


def get_images(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url)
        f.write(response.content)

# Function to scrape Google Images
def scrape_images(class_name, n_img):
    base_url = f"https://www.google.com/search?q={class_name}&tbm=isch"
    driver = webdriver.Chrome()
    driver.get(base_url)
    time.sleep(2)

    for _ in range(15):  # Scroll down 20 times
        driver.execute_script("window.scrollBy(0, 400);")
        time.sleep(2)
        
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    results = soup.find_all('img') 
    print(results)

    count = 0
    icount=0
    isfirst = True
    for img in results:
        if icount == n_img:
            break
        if count > 86 and count%2 !=0 :
            count +=1
        else:
            try:
                if icount==4 and isfirst==True:
                    isfirst=False
                    continue
                img_url = img['src']
                if 'http' in img_url:
                    get_images(img_url, f'{saved_folder}/{class_name}_{icount}.jpg')
                    count += 1
                    icount += 1
            except Exception as e:
                print(f"Failed to download image {count+1}: {e}")
    driver.quit()
    
saved_folder = 'dataset'

# Create the folder if not already existing
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)

# Downloading Kangaroo Images
scrape_images('Kangaroo', 100)

# Downloading Sheep Images
scrape_images('Sheep', 100)