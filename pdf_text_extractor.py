#!/usr/bin/env python3

# Standard library imports
import os
import sys
import io
import logging
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

# Third-party image processing imports
import cv2
import numpy as np
from PIL import Image
import pytesseract

# PDF processing imports
import pymupdf

# Rich CLI interface imports
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)
from rich.console import Console

# Optional GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logging.info("GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    logging.info("GPU acceleration not available")

# Initialize Rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler()
    ]
)

# Constants
MAX_IMAGE_SIZE = 1024
TESSERACT_COMMON_PATHS = [
    '/usr/local/bin/tesseract',
    '/usr/bin/tesseract',
    '/opt/homebrew/bin/tesseract',
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
]
MIN_CONFIDENCE_THRESHOLD = 60.0
LRU_CACHE_SIZE = 100

def gpu_preprocess_image(image):
    """
    GPU-accelerated image preprocessing if CUDA is available
    """
    if GPU_AVAILABLE:
        try:
            # Convert image to numpy array
            img_array = np.array(image)
            
            # Move to GPU
            gpu_img = cp.asarray(img_array)
            
            # Convert to grayscale on GPU
            gpu_gray = cp.dot(gpu_img[...,:3], cp.array([0.299, 0.587, 0.114]))
            
            # Threshold on GPU
            _, gpu_binary = cp.threshold(gpu_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to numpy and create PIL image
            binary_img = cp.asnumpy(gpu_binary).astype(np.uint8)
            return Image.fromarray(binary_img)
        except Exception as e:
            logging.warning(f"GPU processing failed, falling back to CPU: {e}")
    
    # Fallback to CPU preprocessing
    return preprocess_image(image)

def preprocess_image(image):
    """
    Optimize image for faster and more accurate OCR

    Args:
        image (PIL.Image): Input image

    Returns:
        PIL.Image: Preprocessed image
    """
    # Resize large images
    max_size = 1024
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size/image.width, max_size/image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    # Convert to numpy for OpenCV processing
    img_array = np.array(image)

    # Check if image is already grayscale
    if len(img_array.shape) == 2:
        gray = img_array
    else:
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding for better text detection
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(binary)

    return Image.fromarray(denoised)

def image_contains_text(image_bytes, min_confidence_threshold=60.0):
    """
    Determine if an image contains text using more robust detection
    """
    if not is_tesseract_installed():
        return False

    try:
        # Open and preprocess the image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        preprocessed_img = preprocess_image(img)

        # Use Tesseract to get detailed OCR data
        details = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)

        # Calculate the average confidence of detected words
        confidences = [conf for conf in details['conf'] if conf != -1]  # Filter out -1 confidence values
        
        if not confidences:
            return False
            
        avg_confidence = sum(confidences) / len(confidences)
        
        # Count words with meaningful text (excluding single characters and special chars)
        meaningful_words = [word for word in details['text'] 
                          if word.strip() and len(word.strip()) > 1]
        
        # Return True if we have meaningful words and good confidence
        return len(meaningful_words) > 0 and avg_confidence > min_confidence_threshold
        
    except Exception as e:
        logging.error(f"Text detection error: {e}")
        return False

@lru_cache(maxsize=100)
def cached_image_text_extraction(image_bytes):
    """
    Cached image text extraction with preprocessing

    Args:
        image_bytes (bytes): Image data

    Returns:
        str: Extracted text
    """
    img = Image.open(io.BytesIO(image_bytes))
    preprocessed_img = preprocess_image(img)
    return extract_text_from_image(preprocessed_img)

def is_tesseract_installed():
    """
    Check if Tesseract OCR is installed and accessible

    Returns:
        bool: True if Tesseract is installed, False otherwise
    """
    try:
        # Try to find Tesseract executable
        tesseract_path = shutil.which('tesseract')
        if tesseract_path:
            return True

        # Additional check for common Tesseract locations
        common_paths = [
            '/usr/local/bin/tesseract',
            '/usr/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        ]

        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True

        return False
    except Exception as e:
        logging.error(f"Error checking Tesseract installation: {e}")
        return False

def extract_text_from_image(image):
    """
    Extract text from an image using OCR with enhanced error handling

    Args:
        image (PIL.Image): Image to extract text from

    Returns:
        str: Extracted text or empty string
    """
    # Check Tesseract availability before attempting OCR
    if not is_tesseract_installed():
        logging.warning("Tesseract OCR is not installed. Skipping image text extraction.")
        return ""

    try:
        # Validate image before OCR
        if not image or image.width <= 0 or image.height <= 0:
            logging.warning("Invalid image dimensions")
            return ""

        # Attempt OCR with more robust error handling
        text = pytesseract.image_to_string(image)

        # Check if extracted text is meaningful
        if not text or text.isspace():
            logging.info("No meaningful text extracted from image")
            return ""

        return text
    except pytesseract.TesseractError as te:
        logging.error(f"Tesseract OCR Error: {te}")
    except Exception as e:
        logging.error(f"Unexpected error in image text extraction: {e}")

    return ""

def validate_pdf_images(pdf_path):
    """
    Validate images in PDF and identify those with text
    
    Returns:
        dict: Information about PDF images
    """
    image_stats = {
        'total_images': 0,
        'text_images': 0,
        'extractable_images': 0
    }

    try:
        doc = pymupdf.open(pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            images = page.get_images(full=True)
            
            image_stats['total_images'] += len(images)
            
            for img_info in images:
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    img_data = base_image['image']
                    
                    if img_data and len(img_data) > 0:
                        image_stats['extractable_images'] += 1
                        
                        # Check if image contains text
                        if image_contains_text(img_data):
                            image_stats['text_images'] += 1
                
                except Exception:
                    pass
        
        doc.close()
    except Exception as e:
        logging.error(f"PDF image validation error: {e}")
    
    return image_stats

def _process_page(doc_path, page_num, skip_images):
    """
    Process a single page with optimized image text extraction
    """
    # Reopen the document for each process
    doc = pymupdf.open(doc_path)
    page = doc[page_num]
    text = page.get_text() or ""

    # Image text extraction
    if not skip_images:
        images = page.get_images(full=True)
        processed_images = set()  # Track processed images to avoid duplicates

        for img_info in images:
            try:
                xref = img_info[0]

                # Skip if this image has already been processed
                if xref in processed_images:
                    continue

                base_image = doc.extract_image(xref)
                img_data = base_image['image']

                # Only process if image likely contains text
                if image_contains_text(img_data):
                    # Use cached image text extraction
                    image_text = cached_image_text_extraction(img_data)
                    if image_text.strip():
                        text += f"\n\n[Image Text]\n{image_text}"

                processed_images.add(xref)

            except Exception as img_error:
                logging.error(f"Image processing error on page {page_num + 1}: {img_error}")

    doc.close()
    return text

def process_single_pdf_parallel(pdf_path, output_path, skip_images):
    """
    Process a single PDF with rich progress tracking
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn()
    ) as progress:
        # Validate PDF images first
        image_stats = validate_pdf_images(pdf_path) if not skip_images else None
        
        # Overall PDF processing task
        pdf_task = progress.add_task(f"[cyan]Processing PDF: {os.path.basename(pdf_path)}", total=100)
        
        # Log image statistics if available
        if image_stats:
            console.print(f"[bold]PDF Image Stats:[/bold]")
            console.print(f"Total Images: {image_stats['total_images']}")
            console.print(f"Extractable Images: {image_stats['extractable_images']}")
            console.print(f"Images with Text: {image_stats['text_images']}")
        
        try:
            doc = pymupdf.open(pdf_path)
            total_pages = len(doc)
            doc.close()  # Close the document immediately after getting page count

            logging.info(f"Processing PDF: {os.path.basename(pdf_path)} with {total_pages} pages")

            # Collect page texts
            page_texts = []
            page_texts.append(f"--- PDF: {os.path.basename(pdf_path)} ---\n\n")

            # Parallel page processing
            with ProcessPoolExecutor() as executor:
                page_futures = {
                    executor.submit(_process_page, pdf_path, page_num, skip_images): page_num
                    for page_num in range(total_pages)
                }

                for future in as_completed(page_futures):
                    page_num = page_futures[future]
                    try:
                        page_text = future.result()
                        page_texts.append(f"--- Page {page_num + 1} Text Content ---\n")
                        page_texts.append(page_text + '\n\n')
                        
                        # Update progress
                        progress.update(pdf_task, advance=(100/total_pages))
                    except Exception as page_error:
                        logging.error(f"Error processing page {page_num + 1}: {page_error}")

            # Write collected texts to file
            with open(output_path, 'a', encoding='utf-8') as output_file:
                output_file.writelines(page_texts)

            progress.update(pdf_task, completed=100)

        except Exception as e:
            logging.error(f"Critical error during PDF processing: {e}")
            raise

def extract_pdf_text(input_path, output_dir=None, skip_images=False):
    """
    Extract text from a PDF file or all PDFs in a directory with comprehensive compatibility check
    """
    # Prompt for input path if not provided
    while not input_path or not os.path.exists(input_path):
        if input_path:
            logging.warning(f"Path not found: {input_path}")
        input_path = input("Please enter the path to a PDF file or directory: ").strip()

    # Determine output directory
    if output_dir is None:
        output_dir = input("Enter directory to save extracted text (press Enter for current directory): ").strip()
        output_dir = output_dir or os.getcwd()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if input is a directory or a single file
    if os.path.isdir(input_path):
        # Process all PDFs in the directory
        pdf_files = [f for f in os.listdir(input_path) if f.lower().endswith('.pdf')]

        if not pdf_files:
            logging.warning("No PDF files found in the directory")
            return

        # Consolidated output file for all PDFs
        output_path = os.path.join(output_dir, 'consolidated_pdf_text.txt')

        # Parallel PDF processing
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_single_pdf_parallel,
                                os.path.join(input_path, pdf_filename),
                                output_path,
                                skip_images)
                for pdf_filename in pdf_files
            ]

            # Wait for all PDFs to be processed
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"PDF processing error: {e}")

        logging.info(f"Text extracted successfully to {output_path}")
        logging.info(f"Total PDFs processed: {len(pdf_files)}")

    else:
        # Process single PDF file
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '.txt')

        with open(output_path, 'w', encoding='utf-8') as output_file:
            process_single_pdf_parallel(input_path, output_path, skip_images)

        logging.info(f"Text extracted successfully to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract text from PDF files or directories')
    parser.add_argument('input_path', nargs='?', default=None, help='Path to the PDF file or directory')
    parser.add_argument('-o', '--output', help='Directory to save extracted text', default=None)
    parser.add_argument('--skip-images', action='store_true', help='Skip image text extraction')

    args = parser.parse_args()

    try:
        extract_pdf_text(args.input_path, args.output, args.skip_images)
    except Exception as e:
        console.print(f"[bold red]Extraction Failed:[/bold red] {e}")
        logging.error(f"Extraction failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
