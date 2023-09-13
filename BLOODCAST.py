"""""""""
_  _ ____ ___  ____    ___  _   _    ____ _  _ _  _ _  _ ____ ____ 
|\/| |__| |  \ |___    |__]  \_/     |__|  \/  |\ | |__| |___ |__/ 
|  | |  | |__/ |___    |__]   |      |  | _/\_ | \| |  | |___ |  \ 
                                                                   
"""""""""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import newspaper
import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import tensorflow as tf

def load_dataset(true_data_path, false_data_path):
    true_data = []
    with open(true_data_path, 'r', encoding='ISO-8859-1') as file:
        csv_reader = csv.reader(file, delimiter=";")
        for row in csv_reader:
            if len(row) >= 2:
                text = row[1].strip().replace("â", "").replace("â", "").replace("â", "'").replace("â", "-")
                true_data.append([row[0], text, 1])

    false_data = []
    with open(false_data_path, 'r', encoding='ISO-8859-1') as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            if len(row) >= 2:
                text = row[1].strip().replace("â", "").replace("â", "").replace("â", "'").replace("â", "-")
                false_data.append([row[0], text, 0])

    true_df = pd.DataFrame(true_data, columns=["ID", "text", "label"])
    false_df = pd.DataFrame(false_data, columns=["ID", "text", "label"])

    data_df = pd.concat([true_df, false_df])
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    text_data = data_df["text"]
    labels = data_df["label"]

    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(text_data)

    indices = features.nonzero()
    values = features.data

    features = tf.sparse.SparseTensor(indices=np.column_stack(indices),
                                      values=tf.cast(values, tf.float32),
                                      dense_shape=features.shape)

    features = tf.sparse.reorder(features)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)

    return features, labels, vectorizer, label_encoder


def predict_text_classification(new_data, features, vectorizer, label_encoder, model):
    new_text_data = [new_data]
    new_features = vectorizer.transform(new_text_data)

    new_indices = new_features.nonzero()
    new_values = new_features.data

    new_features = tf.sparse.SparseTensor(indices=np.column_stack(new_indices),
                                          values=tf.cast(new_values, tf.float32),
                                          dense_shape=new_features.shape)

    new_features = tf.sparse.reorder(new_features)

    prediction = model.predict(new_features)
    predicted_label = label_encoder.inverse_transform([round(prediction[0][0])])

    return predicted_label

def open_website_and_close_tab(url, driver, duration):
    driver.execute_script("window.open('" + url + "', '_blank')")
    time.sleep(1)
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(duration)
    driver.close()
    if len(driver.window_handles) > 1:
        driver.switch_to.window(driver.window_handles[-1])
    else:
        driver.switch_to.window(driver.window_handles[0])
    return driver

def gatherer():
    # List of news sites with skip counts and updated links
    sites = [
        ("https://www.washingtonpost.com/world/", "Washington Post", 10),
        ("https://www.infowars.com/category/14/", "Infowars", 1),
        ("https://www.foxnews.com/world", "Fox News", 10),
        ("https://www.msnbc.com/", "MSNBC", 10),
        ("https://www.breitbart.com/world-news/", "Breitbart", 10),
        ("https://edition.cnn.com/world", "CNN", 10),
        ("https://theguardian.com/", "The Guardian", 10),
        ("https://www.nytimes.com/section/world", "New York Times", 10)
    ]

    # Initialize the file to write the combined scraped data
    with open("chachacha.txt", "w", encoding="utf-8") as chachacha_file:
        # Scrape articles from each site
        for site_url, site_name, skip_count in sites:
            try:
                site_paper = newspaper.build(site_url, memoize_articles=False)

                # Skip the specified number of articles
                articles = site_paper.articles[skip_count: skip_count + 5]

                # Initialize a file to write the scraped data for the current site
                filename = f"{site_name.replace(' ', '')}.txt"
                with open(filename, "w", encoding="utf-8") as site_file:
                    for article in articles:
                        try:
                            article.download()
                            article.parse()
                            text = article.text.replace("\n", " ")
                            site_file.write(f"Site: {site_name}\n")
                            site_file.write(f"Link: {article.url}\n")
                            site_file.write(f"Text: {text}\n\n")
                            chachacha_file.write(f"Site: {site_name}\n")
                            chachacha_file.write(f"Link: {article.url}\n")
                            chachacha_file.write(f"Text: {text}\n\n")
                            time.sleep(1)  # Introduce a delay of 1 second between requests
                        except Exception as e:
                            print(f"Error parsing article: {str(e)}")

                print(f"Scraping complete for '{site_name}'. The data has been written to '{filename}'.")

            except Exception as e:
                print(f"Error scraping site '{site_name}': {str(e)}")

    print("All scraping operations complete.")

def install_adblock_extension_chrome(extension_path):
    options = Options()
    options.add_extension(extension_path)
    driver = webdriver.Chrome(service=Service(executable_path=r"D:\Bloodcast\chromedriver_win32\chromedriver.exe"), options=options)
    return driver


# START OF PROGRAM #


# Increase the field size limit
csv.field_size_limit(500 * 1024 * 1024)  # Set a larger limit if needed

start_time = time.time()
# Load the dataset
true_data_path = r'D:\Bloodcast\archive\DataSet_Misinfo_TRUE.csv'
false_data_path = r'D:\Bloodcast\archive\DataSet_Misinfo_FAKE.csv'

features, labels, vectorizer, label_encoder = load_dataset(true_data_path, false_data_path)

# Load the trained model
model = load_model('model.h5')

print("Model loaded in", time.time() - start_time, "sec")

# Provide the path to the ad blocker extension (CRX file)
extension_path = r'D:\Bloodcast\adblock-plus-crx-master\bin\Adblock-Plus_v1.12.4.crx'

# Initialize Chrome driver with ad blocker extension
driver = install_adblock_extension_chrome(extension_path)

# Open "Black" in the first tab
driver.get("D:\Bloodcast\Imgforbrowser\Black.png")

# Main cycle
while True:

    # Parser 2
    driver.execute_script("window.open('" + "D:/Bloodcast/Imgforbrowser/Main.png" + "', '_blank')")
    time.sleep(1)
    driver.switch_to.window(driver.window_handles[-1])
    start_time = time.time()
    gatherer()
    print("parser took", time.time() - start_time, "sec to run")
    driver.close()
    if len(driver.window_handles) > 1:
        driver.switch_to.window(driver.window_handles[-1])
    else:
        driver.switch_to.window(driver.window_handles[0])

    # List of document file names
    document_files = [
        'CNN.txt',
        'FoxNews.txt',
        'NewYorkTimes.txt',
        'Breitbart.txt',
        'TheGuardian.txt',
        'Infowars.txt',
        'WashingtonPost.txt',
        'MSNBC.txt'
    ]

    # Determine the maximum number of links in any document
    max_links = 0
    for document_file in document_files:
        with open(document_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            num_links = sum(1 for line in lines if line.startswith("Link: "))
            max_links = max(max_links, num_links)

    # Iterate through the links and texts
    for link_index in range(max_links):
        for document_file in document_files:
            with open(document_file, "r", encoding="utf-8") as file:
                lines = file.readlines()
            links = [line.strip().replace("Link: ", "") for line in lines if line.startswith("Link: ")]
            texts = [line.strip().replace("Text: ", "") for line in lines if line.startswith("Text: ")]
            if link_index < len(links):
                link = links[link_index]
                predicted_label = predict_text_classification(texts[link_index], features, vectorizer, label_encoder, model)
                print(f"Document: {document_file}, Link: {link}, Lable: {predicted_label}")
                try:
                    driver = open_website_and_close_tab(link, driver, 2)  # Open for 20 seconds
                except Exception as e:
                    print("Error occurred:", str(e))
        # Open "fun.com" website for 10 seconds
        driver = open_website_and_close_tab("D:/Bloodcast/Imgforbrowser/Disclamer.png", driver, 3) # Open for 10 seconds

# Finally, when you're done, you can quit the driver
print('End')
