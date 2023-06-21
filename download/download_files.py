#this script downloads and unzips dbnl

def download_and_unzip(url, output_dir = "."):
    import urllib.request
    import zipfile
    # Download the zipfile
    filename, _ = urllib.request.urlretrieve(url)

    # Unzip the zipfile
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    return 

def download(url, output):
    import requests
    response = requests.get(url)
    with open(output, "wb") as file:
        file.write(response.content)
    return