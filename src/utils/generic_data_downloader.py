import os
import requests
import urllib.parse
import urllib.request

def list_directory(base_url):
    response = requests.get(base_url)
    if response.status_code != 200:
        return []
    lines = response.text.splitlines()
    links = []
    for line in lines:
        if 'href="' in line:
            start = line.find('href="') + 6
            end = line.find('"', start)
            href = line[start:end]
            if href not in ('../', './'):
                links.append(href)
    return links

def download_directory(base_url, local_path):
    os.makedirs(local_path, exist_ok=True)
    items = list_directory(base_url)
    
    for item in items:
        full_url = urllib.parse.urljoin(base_url, item)
        local_file_path = os.path.join(local_path, item)
        
        if item.endswith('/'):
            os.makedirs(local_file_path, exist_ok=True)
            download_directory(full_url, local_file_path)
        else:
            urllib.request.urlretrieve(full_url, local_file_path)
            print(f"Downloaded: {local_file_path}")
            
# Example Usage
base_url = "https://www.predsci.com/~pete/research/reza/ADAPT-GONG/"  # Change to your URL
local_path = "pfss"  # Change to your preferred local path
download_directory(base_url, local_path)
