from keras.preprocessing.image import ImageDataGenerator
import numpy as np

data_dir = 'dataset/'
Augmentor = ImageDataGenerator(rescale=1.0/255.0)
test_itr = Augmentor.flow_from_directory(data_dir+'test/', class_mode='binary', batch_size=5, target_size=(200,200),shuffle=False)

# Print the image filenames and true labels for all batches
for i in range(len(test_itr)):
    # Load the next batch of test images and labels
    test_imgs, test_labels = next(test_itr)
    
    # Get the filenames of the test images in the current batch
    batch_filenames = test_itr.filenames[(test_itr.batch_index-1) * test_itr.batch_size : (test_itr.batch_index) * test_itr.batch_size]


    # Print the filenames and corresponding true labels
    for filename, label in zip(batch_filenames, test_labels):
        label_name = 'Kangaroo' if label == 0 else 'Sheep'
        print(f"Filename: {filename}, True Label: {label_name} ({label})")



# # Importing files
# import time
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# import requests
# import os
# from bs4 import BeautifulSoup


# def get_images(url, filename):
#     with open(filename, 'wb') as f:
#         response = requests.get(url)
#         f.write(response.content)

# # Function to scrape Google Images
# def scrape_images(class_name, n_img):
#     base_url = f"https://www.google.com/search?q={class_name}&tbm=isch"
#     driver = webdriver.Chrome()
#     driver.get(base_url)
#     time.sleep(2)

#     for _ in range(15):  # Scroll down 20 times
#         driver.execute_script("window.scrollBy(0, 400);")
#         time.sleep(2)
        
#     soup = BeautifulSoup(driver.page_source, 'html.parser')
#     results = soup.find_all('img') 
#     print(results)

#     count = 0
#     icount=0
#     isfirst = True
#     for img in results:
#         if icount == n_img:
#             break
#         if count > 86 and count%2 !=0 :
#             count +=1
#         else:
#             try:
#                 if icount==4 and isfirst==True:
#                     isfirst=False
#                     continue
#                 img_url = img['src']
#                 if 'http' in img_url:
#                     get_images(img_url, f'{saved_folder}/{class_name}_{icount}.jpg')
#                     count += 1
#                     icount += 1
#             except Exception as e:
#                 print(f"Failed to download image {count+1}: {e}")
#     driver.quit()
    
# saved_folder = 'try'

# # Create the folder if not already existing
# if not os.path.exists(saved_folder):
#     os.makedirs(saved_folder)

# # Downloading Kangaroo Images
# scrape_images('Kangaroo', 120)

# # Downloading Sheep Images
# scrape_images('Sheep', 120)