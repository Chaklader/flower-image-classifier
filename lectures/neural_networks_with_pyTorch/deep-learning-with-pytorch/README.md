$ brew install unzip && wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

$ unzip kagglecatsanddogs_5340.zip

## Dataset Setup

To set up the Cat vs Dog dataset:
1. Run `./setup_dataset.sh`
2. This will download, extract and organize the images
3. The script will clean up temporary files automatically

Note: The dataset is ~800MB and requires internet connection for download
