import datetime
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from loguru import logger

from config.config import Config

# Config loguru logger to log to console and file
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger.add(
    os.path.join(Config.LOGGING_DIR, f"pdf_extractor_azure_{timestamp}.log"),
    level=Config.LOGGING_LEVEL,
    rotation="1 week",
    retention="1 month",
)

load_dotenv()


class PDFExtractorAzure(object):
    """
    Use Azure AI Doc Intelligence Services to extract text and tables from a PDF file.

    Args:
        input_dir (str): Relative path to the raw PDF files.
        output_dir (str): Relative path to the output directory.
    """

    def __init__(self, input_dir: str, output_dir: str) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self) -> None:
        """
        Bulk process all PDF files in the input directory using process_pdf method.
        """
        for file in sorted(os.listdir(self.input_dir)):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(self.input_dir, file)
                self.extract(file_path)

    def extract(self, file_path: str) -> None:
        """
        Process a single PDF file. Output the parsed markdown file in the output directory named after the PDF file.
        Args:
            file_path (str): Relative path to the PDF file
        """
        logger.info(
            f"=============================Processing file: {file_path}============================="
        )

        # TODO: debug "ModuleNotFoundError: No module named 'azure.ai'; 'azure' is not a package" (works in notebook)
        loader = AzureAIDocumentIntelligenceLoader(
            api_endpoint=os.getenv("AZURE_DI_ENDPOINT"),
            api_key=os.getenv("AZURE_DI_API_KEY"),
            file_path=file_path,
            api_model="prebuilt-layout",
        )

        logger.info("Loading document(s)...")
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} document(s).")

        # Write the parsed markdown to the output directory with " " replaced by "_"
        logger.info("Writing parsed markdown to output directory...")
        output_file = os.path.join(
            self.output_dir,
            os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_") + ".md",
        )
        with open(output_file, "w") as f:
            for doc in documents:
                f.write(doc.page_content)


if __name__ == "__main__":
    pdf_w_ground_truth = [
        # "data/raw/mvt_zinc/reports_processed/Bleiberg Pb Zn 5-2017.pdf",
        # "data/raw/mvt_zinc/reports_processed/Bongará Zn 3-2019.pdf",
        "/home/yixin0829/minmod/minmod-poc/data/raw/mvt_zinc/reports_failed/Daniels Harbour Zn 12-2017.pdf",
        # "data/raw/mvt_zinc/reports_processed/Hakkari Zn 3-2010.pdf",
        # "data/raw/mvt_zinc/reports_processed/Hakkari Zn 7-2013.pdf",
        # "data/raw/mvt_zinc/reports_processed/Hakkari Zn 4-2011.pdf",
        # "data/raw/mvt_zinc/reports_processed/Mehdiabad Zn 3-2005.pdf",
        # "data/raw/mvt_zinc/reports_failed/Prairie Creek Zn Pb Ag 9-2017 FS.pdf",
        # "data/raw/mvt_zinc/reports_processed/Reocin Zn 3-2002.pdf",
    ]
    pdf_extractor = PDFExtractorAzure(
        Config.RAW_REPORTS_DIR, Config.PARSED_PDF_DIR_AZURE
    )

    for pdf in pdf_w_ground_truth:
        pdf_extractor.extract(pdf)
