# 📄 PDF Text Extractor (v0.1.0)

## 🌟 Overview

A powerful Python script designed to extract text from PDF files with advanced capabilities, including OCR-based image text extraction and comprehensive error handling. The tool supports both single PDF files and batch processing of entire directories, with optional GPU acceleration for improved performance.

## 🔄 Version

**Current Version:** 0.1.0

**Changes in this version:**
- Initial release with core functionality
- GPU acceleration support
- Parallel processing implementation
- Basic OCR capabilities
- Command-line interface
- Logging system

## ✨ Features

- 📖 Extract text from single PDF files or entire directories
- 🖼️ Intelligent OCR-based image text extraction
- 🚀 GPU acceleration support (optional)
- 🔄 Parallel processing for improved performance
- 🛡️ Comprehensive error handling and validation
- 📊 Progress tracking with rich CLI interface
- 📝 Detailed logging system
- 💾 Caching system for improved performance
- 🔧 Configurable output options

## 🔧 Prerequisites

- 🐍 Python 3.7+
- 🤖 Tesseract OCR installed (required for image text extraction)
- 📦 Required Python packages (see Installation section)
- 🎮 CUDA toolkit (optional, for GPU acceleration)

## 📥 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pdf-text-extractor.git
   cd pdf-text-extractor
   ```

2. **Install Tesseract OCR:**
   - **Ubuntu/Debian:**
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - **macOS:**
     ```bash
     brew install tesseract
     ```
   - **Windows:**
     Download and install from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Install CUDA toolkit for GPU acceleration**
   - Download and install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## 🚀 Usage

### Basic Usage

```bash
python pdf_text_extractor.py input.pdf
```

### Advanced Options

```bash
# Process entire directory
python pdf_text_extractor.py /path/to/pdf/directory

# Specify output directory
python pdf_text_extractor.py input.pdf -o /path/to/output

# Skip image text extraction
python pdf_text_extractor.py input.pdf --skip-images
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `input_path` | Path to PDF file or directory |
| `-o, --output` | Output directory (optional) |
| `--skip-images` | Skip image text extraction |

## 📋 Output Format

The extracted text is saved in the following format:
- Single PDF: `{pdf_name}.txt`
- Directory: `consolidated_pdf_text.txt`

Each file contains:
- PDF metadata
- Page-by-page text content
