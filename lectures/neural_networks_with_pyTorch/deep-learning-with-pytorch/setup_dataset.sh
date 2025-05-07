#!/bin/bash
# Dataset setup script for Cat vs Dog classification
# Run this script to automatically download and organize the dataset

echo "1. Downloading dataset..."
wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

echo "2. Unzipping dataset..."
unzip kagglecatsanddogs_5340.zip -d Cat_Dog_data

echo "3. Organizing dataset using Python script..."
python3 organize_dataset.py

echo "4. Cleaning up..."
rm kagglecatsanddogs_5340.zip
rm -rf Cat_Dog_data/PetImages

echo "Dataset setup complete! Files organized in Cat_Dog_data/{train,test}"
