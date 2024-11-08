import os
import pdfreader
import json

import argparse

def parse_pdfs(pdf_folder_path, output_name):
    # Initialize storage for parsed data
    parsed_data = {}

    # Loop through each PDF in the folder
    for pdf_file in os.listdir(pdf_folder_path):
        if pdf_file.endswith('.pdf'):
            # Set up file path and PDF reader
            pdf_path = os.path.join(pdf_folder_path, pdf_file)
            with open(pdf_path, 'rb') as file:
                reader = pdfreader.PdfFileReader(file)
                text_content = []

                # Extract text from each page
                for page_num in range(reader.numPages):
                    page = reader.getPage(page_num)
                    text_content.append(page.extract_text())

                # Store extracted text under the file name
                parsed_data[pdf_file] = {
                    "text": "\n".join(text_content),
                    "num_pages": reader.numPages
                }

    # Save parsed data to JSON for easy access
    with open(os.path.join(pdf_folder_path, output_name), 'w') as json_file:
        json.dump(parsed_data, json_file)

    print(f"PDF parsing complete. Parsed data saved to {output_name}.")
    return parsed_data

def main():
    parser = argparse.ArgumentParser(description='Parse PDFs in a folder and save extracted text to a JSON file.')
    parser.add_argument('--pdf_folder_path', type=str, help='Path to the '
                                                           'folder containing PDFs to parse.')
    args = parser.parse_args()

    articles = parse_pdfs(args.pdf_folder_path, 'parsed_course_data.json' )

    print('Parsed data:')

if __name__ == '__main__':
    main()