#!/usr/bin/env python3
import os
import sys
import subprocess
import webbrowser
import time
import re
import gdown
import glob

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
PROCESSED_DIR = os.path.join(CURRENT_DIR, 'processed_data')

# Required files
REQUIRED_FILES = [
    "BIGNEWSBLN_center.json",
    "BIGNEWSBLN_left.json",
    "BIGNEWSBLN_right.json"
]

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def open_form():
    """Open the Google form for the user to fill out."""
    form_url = "https://forms.gle/yRx5ANHKNuj1kgBDA"
    
    print("\n" + "="*80)
    print("BIGNEWSBLN Dataset Download Script")
    print("="*80)
    print("\nTo download the BIGNEWSBLN dataset, you need to fill out a Google form first.")
    print("This script will open the form in your default web browser.")
    print("\nIMPORTANT: When filling out the form, make sure to select 'BIGNEWSBLN' dataset option.")
    print("BIGNEWSBLN is the balanced version of BIGNEWS with equal number of articles from left/center/right.")
    print("\nAfter submitting the form, you will receive a Google Drive link.")
    print("You will need to copy that link and paste it back into this script.")
    print("\nOpening form now...")
    
    try:
        # Try to open the URL in the default web browser
        webbrowser.open(form_url)
        print(f"\nIf the browser didn't open automatically, please visit this URL manually:")
        print(f"{form_url}")
    except Exception as e:
        print(f"\nFailed to open the browser automatically. Please visit this URL manually:")
        print(f"{form_url}")
        print(f"Error: {e}")
    
    print("\n" + "="*80)
    input("Press Enter after you've submitted the form and received the download link for BIGNEWSBLN dataset...")

def get_drive_link():
    """Get the Google Drive link from the user."""
    print("\n" + "="*80)
    print("Please paste the Google Drive link for the BIGNEWSBLN dataset:")
    print("(Make sure you selected 'BIGNEWSBLN' option in the form)")
    drive_link = input("> ").strip()
    
    while not drive_link or not (drive_link.startswith("https://drive.google.com") or 
                                drive_link.startswith("http://drive.google.com")):
        print("\nInvalid link. The link should start with 'https://drive.google.com'.")
        print("Please paste the correct Google Drive link for BIGNEWSBLN dataset:")
        drive_link = input("> ").strip()
    
    return drive_link

def extract_folder_id(drive_link):
    """Extract the folder ID from a Google Drive link."""
    # Pattern for folder ID in Google Drive URLs
    patterns = [
        r"https://drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/drive/u/\d+/folders/([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_link)
        if match:
            return match.group(1)
    
    return None

def download_from_drive(folder_id):
    """Download all files from a Google Drive folder."""
    print("\n" + "="*80)
    print(f"Downloading BIGNEWSBLN dataset from Google Drive folder ID: {folder_id}")
    print("This may take some time depending on the size of the files...")
    
    try:
        # Change to the data directory
        os.chdir(DATA_DIR)
        
        # Download the entire folder
        gdown.download_folder(id=folder_id, quiet=False, use_cookies=False)
        
        print("\nDownload completed. Verifying files...")
        
        # Verify that all required files are present
        all_files_found = True
        missing_files = []
        
        # Search for the required files in the data directory and its subdirectories
        for required_file in REQUIRED_FILES:
            # Use glob to find the file anywhere in the data directory
            matches = glob.glob(f"**/{required_file}", recursive=True)
            if not matches:
                all_files_found = False
                missing_files.append(required_file)
        
        if not all_files_found:
            print("\nWARNING: Some required files are missing from the downloaded dataset:")
            for missing_file in missing_files:
                print(f"  - {missing_file}")
            print("\nPlease make sure you selected the BIGNEWSBLN dataset in the form.")
            print("The download may be incomplete or incorrect.")
            return False
        
        print("\nAll required files found. Download successful!")
        return True
    except Exception as e:
        print(f"\nError downloading files: {e}")
        print("Please check your internet connection and the Google Drive link.")
        return False
    finally:
        # Change back to the original directory
        os.chdir(CURRENT_DIR)

def main():
    """Main function to guide the user through the download process."""
    # Step 1: Open the form for the user to fill out
    open_form()
    
    # Step 2: Get the Google Drive link from the user
    drive_link = get_drive_link()
    
    # Step 3: Extract the folder ID from the link
    folder_id = extract_folder_id(drive_link)
    
    if not folder_id:
        print("\nCould not extract a valid folder ID from the provided link.")
        print("Please make sure you've copied the correct Google Drive link.")
        sys.exit(1)
    
    print(f"\nExtracted folder ID: {folder_id}")
    
    # Step 4: Download all files from the Google Drive folder
    success = download_from_drive(folder_id)
    
    if success:
        print("\n" + "="*80)
        print("BIGNEWSBLN dataset downloaded successfully!")
        print(f"Files have been downloaded to: {DATA_DIR}")
        print("\nThe dataset includes the following files:")
        for file in REQUIRED_FILES:
            print(f"  - {file}")
        print("\nYou can now process the data as needed.")
    else:
        print("\n" + "="*80)
        print("Download failed or dataset incomplete.")
        print("Please try again and make sure to select the BIGNEWSBLN dataset in the form.")
    
    print("="*80)

if __name__ == "__main__":
    main() 