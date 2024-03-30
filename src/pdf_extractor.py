import csv
import datetime
import json
import logging
import os.path
import re

import pandas as pd
from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.client_config import ClientConfig
from adobe.pdfservices.operation.exception.exceptions import (
    SdkException,
    ServiceApiException,
    ServiceUsageException,
)
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import (
    ExtractElementType,
)
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import (
    ExtractPDFOptions,
)
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_renditions_element_type import (
    ExtractRenditionsElementType,
)
from adobe.pdfservices.operation.pdfops.options.extractpdf.table_structure_type import (
    TableStructureType,
)
from dotenv import load_dotenv

from config.config import Config
from utils.utils import convert_to_numeric, normalize_unicode

load_dotenv()

# Create 'log/' directory if it doesn't exist
log_directory = Config.LOGGING_DIR
os.makedirs(log_directory, exist_ok=True)
# Generate the log filename with the current timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"pdf_extractor_{timestamp}.log"
log_path = os.path.join(log_directory, log_filename)

# Configure logging
logging.basicConfig(
    level=Config.LOGGING_LEVEL,  # Set the logging level to DEBUG (or another appropriate level)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path),  # Log messages to a file
        logging.StreamHandler(),  # Additionally, log messages to stderr (optional)
    ],
)


class PDFExtractor(object):
    """
    Use Adobe PDF Services SDK to extract text and tables from a PDF file.

    Args:
        input_dir (str): Relative path to the raw PDF files.
        output_dir (str): Relative path to the output directory.
    """

    def __init__(self, input_dir: str, output_dir: str) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Initial setup, create credentials instance.
        self.service_credentials = (
            Credentials.service_principal_credentials_builder()
            .with_client_id(os.getenv("PDF_SERVICES_CLIENT_ID"))
            .with_client_secret(os.getenv("PDF_SERVICES_CLIENT_SECRET"))
            .build()
        )

    def run(self) -> None:
        """
        Bulk process all PDF files in the input directory using process_pdf method.
        """
        for file in sorted(os.listdir(self.input_dir)):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(self.input_dir, file)
                self.process_pdf(file_path)

        # Unzip the result.zip files.
        self.unzip_result()

    def process_pdf(self, file_path: str) -> None:
        """
        Process a single PDF file using Adobe PDF Services SDK.
        Output the result to a new directory in the output directory named after the PDF file.
        Args:
            file_path (str): Relative path to the PDF file
        """
        logging.info(
            f"=============================Processing file: {file_path}============================="
        )
        try:
            # Get base path.
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logging.info("Base path: " + base_path)

            # Create client config instance with custom time-outs. (x10 default time-outs)
            client_config = (
                ClientConfig.builder()
                .with_connect_timeout(40000)
                .with_read_timeout(100000)
                .build()
            )

            # Create an ExecutionContext using credentials and create a new operation instance.
            execution_context = ExecutionContext.create(self.service_credentials)
            extract_pdf_operation = ExtractPDFOperation.create_new()

            # Set operation input from a source file.
            source = FileRef.create_from_local_file(os.path.join(base_path, file_path))
            extract_pdf_operation.set_input(source)

            # Build ExtractPDF options and set them into the operation
            extract_pdf_options: ExtractPDFOptions = (
                ExtractPDFOptions.builder()
                .with_elements_to_extract(
                    [ExtractElementType.TEXT, ExtractElementType.TABLES]
                )  # Extract text and tables
                .with_element_to_extract_renditions(
                    ExtractRenditionsElementType.TABLES
                )  # Extract tables as .pn figures
                .with_table_structure_format(
                    TableStructureType.CSV
                )  # Extract tables as CSV
                .build()
            )
            extract_pdf_operation.set_options(extract_pdf_options)

            start = datetime.datetime.now()
            # Execute the operation.
            result: FileRef = extract_pdf_operation.execute(execution_context)

            # Extract PDF name from file_path, replace whitespace with "_", and use it to create a new directory.
            file_name = file_path.split("/")[-1].split(".")[0].replace(" ", "_")
            logging.info("File name: " + file_name)
            # Create a new directory to store the result.
            os.makedirs(
                os.path.join(base_path, self.output_dir, f"{file_name}"), exist_ok=True
            )
            logging.info(
                "Directory created: " + base_path + self.output_dir + f"/{file_name}"
            )
            # Save the result to the specified location and log the elapsed time.
            result.save_as(
                os.path.join(base_path, self.output_dir, f"{file_name}/result.zip")
            )
            logging.info(
                f"Saved result. Elapsed time: {datetime.datetime.now() - start}"
            )

            # TODO: Move the successfully processed file to the processed directory. Right now do it manually.

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception("Exception encountered while executing operation ")

    def unzip_result(self) -> None:
        """
        Loop through all directories in the output dir, extract the result.zip file and delete the zip file.
        """
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file.lower().endswith(".zip"):
                    logging.info("Unzipping file: " + file)
                    file_path = os.path.join(root, file)
                    os.system(f"unzip {file_path} -d {root}")
                    os.remove(file_path)
                    logging.info("Zip file deleted: " + file)


class PDFResponseParser(object):
    """
    Parse the useful result of the PDF extraction. (paragraphs in structured JSON and CSV tables) into a single text file.
    """

    def __init__(self, input_dir, output_dir) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    def _preprocess_text(self, text: str) -> str:
        text = text.strip()
        text = text.replace(" +", " ")
        text = normalize_unicode(text)
        return text

    def _preprocess_table_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip the leading and trailing whitespaces from all cells in df.
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Detect if a string is a number and convert it to a number.
        df = df.map(lambda x: convert_to_numeric(x) if isinstance(x, str) else x)

        # Replace NaN with an empty string
        df = df.fillna("")

        return df

    def run(self, overwrite: bool) -> None:
        """
        Parse the result of the PDF extraction in input dir into text files in output dir.
        Overwrite the existing text files if overwrite is True.
        """
        # Read directories in the input directory.
        _, dirs, _ = next(os.walk(self.input_dir))
        for i, dir in enumerate(sorted(dirs)):
            logging.info(f"Processing directory {i+1}/{len(dirs)}: {dir}")

            # Create a new directory in the output directory with the same name if it doesn't exist.
            # If it exists, skip the directory.
            if os.path.exists(os.path.join(self.output_dir, dir)) and not overwrite:
                logging.info(f"Directory {dir} already exists. Skipping...")
                continue
            else:
                # if overwrite is True, delete the existing directory and create a new one.
                output_path = os.path.join(self.output_dir, dir)
                os.system(f"rm -r {output_path}")
                os.makedirs(output_path)

            # Deserialize the structuredData JSON file and store the paragraphs and tables in a single text file.
            with open(
                os.path.join(self.input_dir, dir, "structuredData.json"), "r"
            ) as f:
                data = json.load(f)

            with open(os.path.join(output_path, f"{dir}.txt"), "a") as result:
                result.write("# COVER PAGE\n\n")
                next_table = False  # Flag to indicate if the next element is a table
                for element in data.get("elements"):
                    logging.debug(element)
                    if element["Path"].startswith("//Document/H"):
                        if element["Page"] > 0:
                            # Match the digit come after the H tag.
                            header_lvl = re.search(r"\d+", element["Path"]).group(0)
                            prefix = "#" * int(header_lvl) + " "
                            if element.get("Text"):
                                preprocessed_text = self._preprocess_text(
                                    element.get("Text")
                                )
                                result.write(prefix + preprocessed_text + "\n")
                        else:
                            # If the header is on the first page, append it to the result.txt file without header.
                            if element.get("Text"):
                                preprocessed_text = self._preprocess_text(
                                    element.get("Text")
                                )
                                result.write(preprocessed_text + "\n")

                    # If the element is a paragraph, append the text to the result.txt file.
                    elif element["Path"].startswith("//Document/P") or element[
                        "Path"
                    ].startswith("//Document/Footnote"):
                        if element.get("Text"):
                            preprocessed_text = self._preprocess_text(
                                element.get("Text")
                            )

                            # If the paragraph is a table title, append it to the result.txt file without new line.
                            if (
                                preprocessed_text.lower().startswith("table")
                                or next_table
                            ):
                                result.write(preprocessed_text + "\n")
                                next_table = True
                            else:
                                result.write(preprocessed_text + "\n\n")

                    # If the element is a list body and contain "Body" in the path
                    elif element["Path"].startswith("//Document/L") and element[
                        "Path"
                    ].endswith("LBody"):
                        if element.get("Text"):
                            preprocessed_text = self._preprocess_text(
                                element.get("Text")
                            )
                            result.write("- " + preprocessed_text + "\n")

                    # If the element is a table, append the corresponding CSV table from csv_tables list to the result.txt file.
                    elif element["Path"].startswith("//Document/Table") and element.get(
                        "filePaths"
                    ):
                        next_table = False
                        if element["filePaths"][0].endswith(".csv"):
                            # Read the CSV file
                            csv_path = os.path.join(
                                self.input_dir, dir, element["filePaths"][0]
                            )
                            with open(csv_path, "r", encoding="utf-8-sig") as csv_file:
                                # Read from the CSV file
                                reader = csv.reader(csv_file, delimiter=",")
                                # Find the maxium row length
                                max_row_length = max(len(row) for row in reader)
                                # Normalize each row to the max row length
                                normalized_rows = []
                                csv_file.seek(0)
                                for row in reader:
                                    normalized_rows.append(
                                        row + [""] * (max_row_length - len(row))
                                    )
                                # Create a DataFrame from the normalized rows
                                df = pd.DataFrame(normalized_rows)

                                # Preprocess the table DataFrame
                                df = self._preprocess_table_df(df)

                                # Write the DataFrame to the result.txt file
                                result.write(
                                    df.to_csv(index=False, header=False) + "\n"
                                )


if __name__ == "__main__":
    # Extract text and tables from a PDF file.
    # pdf_processor = PDFExtractor(Config.RAW_REPORTS_DIR, Config.PDF_EXTRACTION_DIR)
    # pdf_processor.process_pdf(
    #     file_path="/home/yixin0829/minmod/minmod-poc/data/raw/mvt_zinc/reports_failed/Prairie Creek Zn Pb Ag 9-2017 FS.pdf"
    # )

    # Parse the result of the PDF extraction.
    parser = PDFResponseParser(Config.PDF_EXTRACTION_DIR, Config.PARSED_PDF_DIR)
    parser.run(overwrite=True)
