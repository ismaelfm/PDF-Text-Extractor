## PDF Text Extractor Function Documentation

This document describes the functions within the `pdf_text_extractor.py` script, their purpose, the libraries they use, and why. It also outlines the workflow when the `main` function is called.

### 1. `gpu_preprocess_image(image)`

* **Purpose:** Preprocesses an image for OCR, using GPU acceleration if available (CUDA).
* **Libraries:** `cupy` (for GPU processing), `numpy`, `cv2` (OpenCV), `PIL` (Pillow)
* **Why:** GPU processing can significantly speed up image preprocessing, especially for large images or batches of images. If a GPU is not available, it falls back to the CPU-based `preprocess_image` function.
* **Workflow:** Converts the image to a NumPy array. If a GPU is available, it moves the array to the GPU, converts to grayscale, applies thresholding, and returns the processed image. Otherwise, it calls `preprocess_image`.

### 2. `preprocess_image(image)`

* **Purpose:** Preprocesses an image for OCR on the CPU.
* **Libraries:** `numpy`, `cv2` (OpenCV), `PIL` (Pillow)
* **Why:** Image preprocessing is crucial for accurate OCR. This function resizes large images, converts to grayscale, and applies thresholding to optimize the image for Tesseract.
* **Workflow:** Resizes the image if it's too large, converts it to grayscale using OpenCV, applies Otsu's thresholding to create a binary image, and returns the processed image.

### 3. `image_contains_text(image_bytes, min_confidence_threshold=0.5)`

* **Purpose:** Efficiently checks if an image contains any text.
* **Libraries:** `pytesseract`, `PIL` (Pillow), `io`
* **Why:** Avoids unnecessary full text extraction if an image doesn't contain text.
* **Workflow:** Opens the image from bytes, preprocesses it, and uses `pytesseract.image_to_data` to get word-level confidence scores. Returns `True` if any words have confidence above the threshold, `False` otherwise.

### 4. `cached_image_text_extraction(image_bytes)`

* **Purpose:** Extracts text from an image using OCR, with caching for performance.
* **Libraries:** `lru_cache` (from `functools`), `PIL` (Pillow), `io`
* **Why:** Caching avoids redundant OCR operations on the same image data, improving performance.
* **Workflow:** Uses `lru_cache` to store the results of text extraction. If the image is already cached, it returns the cached text. Otherwise, it preprocesses the image and calls `extract_text_from_image`.

### 5. `is_tesseract_installed()`

* **Purpose:** Checks if Tesseract OCR is installed on the system.
* **Libraries:** `shutil`, `os`, `pytesseract`
* **Why:** Ensures that Tesseract is available before attempting OCR.
* **Workflow:** Uses `shutil.which('tesseract')` to find the Tesseract executable. If not found, it checks some common installation paths.

### 6. `extract_text_from_image(image)`

* **Purpose:** Extracts text from a preprocessed image using Tesseract OCR.
* **Libraries:** `pytesseract`
* **Why:** Core function for image-based text extraction.
* **Workflow:** Calls `pytesseract.image_to_string` to perform OCR. Includes error handling for invalid images and Tesseract errors.

### 7. `validate_pdf_images(pdf_path)`

* **Purpose:** Analyzes a PDF to determine how many images it contains and how many are likely to contain text.
* **Libraries:** `pymupdf`
* **Why:** Provides statistics about images in the PDF, useful for understanding the document's content and potential OCR processing time.
* **Workflow:** Opens the PDF with `pymupdf`, iterates through each page, extracts images, and checks if they contain text using `image_contains_text`. Returns a dictionary with image statistics.

### 8. `_process_page(doc_path, page_num, skip_images)`

* **Purpose:** Processes a single page of a PDF, extracting text and image text (if not skipped).
* **Libraries:** `pymupdf`
* **Why:** Separates page processing logic for parallel processing.
* **Workflow:** Opens the PDF, gets the page text, extracts images (if `skip_images` is False), checks if images contain text, extracts text from images using `cached_image_text_extraction`, and returns the combined text.

### 9. `process_single_pdf_parallel(pdf_path, output_path, skip_images)`

* **Purpose:** Processes a single PDF file, extracting text from all pages in parallel.
* **Libraries:** `pymupdf`, `concurrent.futures` (for parallel processing), `rich` (for progress bar), `os`
* **Why:** Parallel processing significantly speeds up PDF text extraction.
* **Workflow:** Validates PDF images, creates a progress bar, opens the PDF, submits page processing tasks to a `ProcessPoolExecutor`, collects the results, writes the extracted text to the output file, and updates the progress bar.

### 10. `extract_pdf_text(input_path, output_dir=None, skip_images=False)`

* **Purpose:** Main function to handle PDF text extraction from a file or directory.
* **Libraries:** `os`, `argparse`, `concurrent.futures`
* **Why:** Entry point for the script. Handles both single files and directories.
* **Workflow:** Prompts for input path if not provided, determines output directory, checks if input is a file or directory, processes all PDFs in a directory in parallel or a single PDF, and writes the extracted text to output files.

### 11. `main()`

* **Purpose:** Parses command-line arguments and calls `extract_pdf_text`.
* **Libraries:** `argparse`, `sys`
* **Why:** Handles command-line interface and error handling.
* **Workflow:** Parses arguments using `argparse`, calls `extract_pdf_text` with the provided arguments, and handles any exceptions during extraction.

### Workflow when `main` is called:

1. The script starts by parsing command-line arguments using `argparse`.
2. It then calls the `extract_pdf_text` function with the parsed arguments.
3. `extract_pdf_text` determines whether the input is a single PDF file or a directory.
4. If it's a directory, it processes all PDF files in the directory in parallel using `process_single_pdf_parallel`.
5. If it's a single file, it processes the file using `process_single_pdf_parallel`.
6. `process_single_pdf_parallel` uses multiple processes to extract text from each page of the PDF concurrently.
7. The extracted text from each page, including image text (if not skipped), is collected and written to an output file.
8. The script finishes by printing a success message or an error message if any issues occurred.
