import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import pandas as pd
import re
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import trafilatura
import json
from datetime import datetime
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
import pickle
import faiss
import openai
import dotenv
import corelink

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



import asyncio
import corelink
from corelink.resources.control import subscribe_to_stream
sender_id = None
receiver_id = None
embedder = None

async def callback(data_bytes, streamID, header):
    global sender_id, embedder
    if streamID != sender_id:
        print(f"Stream ID: {streamID}")
        data = data_bytes.decode('utf-8')  # Decode bytes to string
        print(f"Received data: {data}")
        # Start search and answer
        await embedder.interactive_search_and_answer(data)

async def update(response, key):
    print(f'Updating as new sender valid in the workspace: {response}')
    # Optionally subscribe to the new stream
    print('......................')
    await subscribe_to_stream(response['receiverID'], response['streamID'])

async def stale(response, key):
    print(response)

async def subscriber(response, key):

    print("subscriber: ", response)





class WebScraper:
    def __init__(self, base_urls, output_folder, url_file='scraped_urls.json'):
        self.base_urls = base_urls if isinstance(base_urls, list) else [base_urls]
        self.output_folder = output_folder
        self.visited_urls = set()
        self.url_file = url_file
        self.scraped_urls = self.load_scraped_urls()
        self.driver = None
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Retry setup
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def load_scraped_urls(self):
        try:
            with open(self.url_file, 'r') as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()

    def save_scraped_urls(self):
        with open(self.url_file, 'w') as f:
            json.dump(list(self.scraped_urls), f, indent=2)

    def get_page_content(self, url):
        time.sleep(random.uniform(1, 3))
        
        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching {url} with requests: {str(e)}. Trying with Selenium.")
            return self.get_page_content_selenium(url)

    def get_page_content_selenium(self, url):
        if not self.driver:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")
            self.driver = webdriver.Chrome(options=chrome_options)

        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Error fetching {url} with Selenium: {str(e)}")
            return None

    def save_page(self, url, content):
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc.replace(':', '_')
        safe_path = re.sub(r'[<>:"/\\|?*]', '_', parsed_url.path.strip('/'))
        if not safe_path:
            safe_path = 'index'
        file_path = os.path.join(self.output_folder, netloc, safe_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(f"{file_path}.html", 'w', encoding='utf-8') as f:
            f.write(content)

    def scrape_page(self, url, base_url):
        current_time = datetime.now().isoformat()

        if url in self.scraped_urls:
            logger.info(f"Already scraped: {url}")
            return []

        self.visited_urls.add(url)
        logger.info(f"Scraping: {url}")

        content = self.get_page_content(url)

        if not content:
            logger.error(f"Failed to fetch content for {url}")
            return []

        self.save_page(url, content)

        # Update scraped_urls set
        self.scraped_urls.add(url)
        self.save_scraped_urls()

        soup = BeautifulSoup(content, 'html.parser')
        links = soup.find_all('a', href=True)

        new_urls = []
        for link in links:
            new_url = urljoin(url, link['href'])
            if new_url.startswith(base_url) and new_url not in self.visited_urls:
                new_urls.append(new_url)

        return new_urls

    def scrape(self):
        urls_to_scrape = self.base_urls.copy()
        with ThreadPoolExecutor(max_workers=5) as executor:
            while urls_to_scrape:
                future_to_url = {executor.submit(self.scrape_page, url, base_url): url for base_url in self.base_urls for url in urls_to_scrape if url.startswith(base_url)}
                urls_to_scrape = []
                for future in tqdm(future_to_url, desc="Scraping pages"):
                    new_urls = future.result()
                    urls_to_scrape.extend([url for url in new_urls if url not in self.scraped_urls])

        if self.driver:
            self.driver.quit()

class DataCleaner:
    def __init__(self, input_folder, output_file):
        self.input_folder = input_folder
        self.output_file = output_file
    
    @staticmethod
    def extract_main_content(html_content):
        extracted = trafilatura.extract(html_content, include_links=False, include_images=False, include_tables=False)
        if extracted:
            cleaned = re.sub(r'\s+', ' ', extracted).strip()
            cleaned = re.sub(r'\n+', '\n', cleaned)
            return cleaned
        return None

    def get_source_url(self, file_path):
        # Logic to reconstruct source URL from file path
        parts = file_path.split(os.sep)
        netloc = parts[1].replace('_', ':')
        path = '/'.join(parts[2:]).replace('_', '/').replace('.html', '')
        source_url = f"https://{netloc}/{path}"
        return source_url

    def clean_data(self):
        data = []

        for root, dirs, files in os.walk(self.input_folder):
            for file in tqdm(files, desc="Cleaning data"):
                if file.endswith('.html'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                            # Extract main content
                            main_content = self.extract_main_content(content)

                            if main_content:
                                source_url = self.get_source_url(file_path)
                                data.append({
                                    'file': file_path,
                                    'content': main_content,
                                    'source_url': source_url
                                })
                            else:
                                logger.warning(f"No main content extracted from {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")

        df = pd.DataFrame(data)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Cleaned data saved to {self.output_file}")

class DatabaseManager:
    def __init__(self, db_file='metadata.db'):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file TEXT,
            chunk_id INTEGER,
            chunk TEXT,
            source_url TEXT
        )
        ''')
        self.conn.commit()

    def insert_metadata(self, file, chunk_id, chunk, source_url):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO metadata (file, chunk_id, chunk, source_url)
        VALUES (?, ?, ?, ?)
        ''', (file, chunk_id, chunk, source_url))
        self.conn.commit()
        return cursor.lastrowid

    def get_metadata(self, ids):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM metadata WHERE id IN ({seq})'.format(
            seq=','.join(['?']*len(ids))), ids)
        return cursor.fetchall()

    def close(self):
        self.conn.close()

class RAGPreparator:
    def __init__(self, cleaned_data_file, output_file, chunk_size=1000, db_manager=None):
        self.cleaned_data_file = cleaned_data_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.db_manager = db_manager

    def prepare_for_rag(self):
        df = pd.read_csv(self.cleaned_data_file)

        rag_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing RAG data"):
            content = row['content']
            chunks = []
            current_chunk = ""

            sentences = re.split(r'(?<=[.!?])\s+', content)

            for sentence in sentences:
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            for chunk_num, chunk in enumerate(chunks):
                rag_data.append({
                    'file': row['file'],
                    'chunk_id': chunk_num,
                    'chunk': chunk,
                    'source_url': row.get('source_url', '')
                })
                # Insert into database
                if self.db_manager:
                    self.db_manager.insert_metadata(row['file'], chunk_num, chunk, row.get('source_url', ''))

        rag_df = pd.DataFrame(rag_data)
        rag_df.to_csv(self.output_file, index=False)
        logger.info(f"RAG-prepared data saved to {self.output_file}")

class FaissEmbedder:
    def __init__(self, rag_data_file, index_file="faiss_index.pkl", db_manager=None):
        self.rag_data_file = rag_data_file
        self.index_file = index_file
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Or any other model you prefer
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.db_manager = db_manager

    def create_index(self):
        index = faiss.IndexFlatL2(self.dimension)
        return index

    def embed_and_insert(self):
        df = pd.read_csv(self.rag_data_file)
        index = self.create_index()
        ids = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding and inserting"):
            embedding = self.model.encode(row['chunk']).astype('float32')
            index.add(np.array([embedding]))
            # Retrieve the database ID
            if self.db_manager:
                cursor = self.db_manager.conn.cursor()
                cursor.execute('SELECT id FROM metadata WHERE file=? AND chunk_id=?', (row['file'], row['chunk_id']))
                db_row = cursor.fetchone()
                if db_row:
                    ids.append(db_row[0])
                else:
                    logger.error(f"Metadata not found for {row['file']} chunk {row['chunk_id']}")
                    ids.append(-1)  # Placeholder for missing ID
            else:
                ids.append(idx)

        # Save the index and IDs
        with open(self.index_file, 'wb') as f:
            pickle.dump({'index': index, 'ids': ids}, f)

        logger.info(f"Inserted {index.ntotal} entities into FAISS index and saved to {self.index_file}")

    def search(self, query, k=5):
        with open(self.index_file, 'rb') as f:
            data = pickle.load(f)
            index = data['index']
            ids = data['ids']

        query_vector = self.model.encode(query).astype('float32')
        distances, indices = index.search(np.array([query_vector]), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(ids):
                meta = self.db_manager.get_metadata([ids[idx]])[0]
                results.append({
                    'distance': distances[0][i],
                    'metadata': {
                        'file': meta[1],
                        'chunk_id': meta[2],
                        'chunk': meta[3],
                        'source_url': meta[4]
                    }
                })
            else:
                logger.warning(f"Invalid index {idx} during search.")
        return results

    async def generate_answer(self, query, k=5):
        results = self.search(query, k=k)
        context = "\n".join([f"From {result['metadata']['source_url']}:\n{result['metadata']['chunk']}" for result in results])

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]

        # Ensure you have set your OpenAI API key
        openai.api_key = ''

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        answer = response['choices'][0]['message']['content']
        print("\nAnswer:")
        print(answer)
        await corelink.send(sender_id, answer)

    async def interactive_search_and_answer(self, query):
        print(f"Received query: {query}")
        if query.lower() == 'quit':
            print("Thank you for using NYU HSRN Q&A. Goodbye!")
            return

        await self.generate_answer(query)

        print("\nTop 3 relevant chunks:")
        results = self.search(query, k=3)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Relevance Score: {1 / (1 + result['distance']):.2f}")
            print(f"Chunk: {result['metadata']['chunk'][:200]}...")
            print(f"Source URL: {result['metadata']['source_url']}")

async def main():
    global sender_id, embedder
    await corelink.set_data_callback(callback)
    await corelink.set_server_callback(update, 'update')
    await corelink.set_server_callback(stale, 'stale')
    await corelink.set_server_callback(subscriber, 'subscriber')  

    base_urls = [
        "https://k8s-docs.hsrn.nyu.edu/",
        "https://corelink-docs.hsrn.nyu.edu/",
        "https://corelink.hsrn.nyu.edu/"
        # Add more URLs as needed
    ]
    output_folder = "scraped_data"
    cleaned_output = "cleaned_data.csv"
    rag_output = "rag_prepared_data.csv"
    faiss_index_file = "faiss_index.pkl"
    db_file = "metadata.db"

    # Initialize Database Manager
    db_manager = DatabaseManager(db_file)

    # 1: Scrape websites (if needed)
    scraper = WebScraper(base_urls, output_folder, url_file='scraped_urls.json')
    if not os.path.exists(output_folder) or not os.listdir(output_folder):
        logger.info("Starting web scraping...")
        scraper.scrape()
    else:
        logger.info("Scraped data already exists. Skipping scraping step.")

    # 2: Clean data (if needed)
    if not os.path.exists(cleaned_output):
        logger.info("Starting data cleaning...")
        cleaner = DataCleaner(output_folder, cleaned_output)
        cleaner.clean_data()
    else:
        logger.info("Cleaned data already exists. Skipping cleaning step.")

    # 3: Prepare for RAG (if needed)
    if not os.path.exists(rag_output):
        logger.info("Starting RAG preparation...")
        preparator = RAGPreparator(cleaned_output, rag_output, chunk_size=1000, db_manager=db_manager)
        preparator.prepare_for_rag()
    else:
        logger.info("RAG-prepared data already exists. Skipping preparation step.")

    # 4: Embed and insert into FAISS (if needed)
    embedder = FaissEmbedder(rag_output, index_file=faiss_index_file, db_manager=db_manager)
    if not os.path.exists(faiss_index_file):
        logger.info("Starting embedding and insertion into FAISS...")
        embedder.embed_and_insert()
    else:
        logger.info("FAISS index already exists. Skipping embedding and insertion step.")

    logger.info("All preprocessing steps completed.")


    
    await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    
    receiver_id = await corelink.create_receiver("HSRNQ", "ws", alert=True, echo=True)
    sender_id = await corelink.create_sender("HSRNA", "ws", "Q&A")
    print(f'Receiver ID: {receiver_id}')
    
    print("Start receiving")
    await corelink.keep_open()
    try:

        while True:
            await asyncio.sleep(3600)   
    except KeyboardInterrupt:
        print('Receiver terminated.')
        await corelink.close()

    await corelink.send(sender_id, "Welcome to HSRN Q&A. Type 'quit' to exit.")
  

    # Close the database connection when done
    db_manager.close()

corelink.run(main())
