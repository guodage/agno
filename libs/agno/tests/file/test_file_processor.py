import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import UploadFile, HTTPException

from agno.file.file_processor import process_uploaded_files, _extract_file_content


class TestProcessUploadedFiles:
    """Test cases for process_uploaded_files function"""

    @pytest.fixture
    def sample_pdf_path(self):
        """Path to a sample PDF file for testing"""
        return "/Users/jiangjiangdear/Documents/3. SpringAI.pdf"

    @pytest.fixture
    def sample_pdf_file(self, sample_pdf_path):
        """Create a mock UploadFile object for PDF testing"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("Sample PDF file not found")

        with open(sample_pdf_path, "rb") as f:
            content = f.read()

        file = Mock(spec=UploadFile)
        file.filename = "3. SpringAI.pdf"
        file.content_type = "application/pdf"
        file.read = AsyncMock(return_value=content)

        return file

    @pytest.mark.asyncio
    async def test_process_uploaded_files_with_pdf_no_knowledge_base(self, sample_pdf_file):
        """Test processing a PDF file without knowledge base"""
        # Act
        direct_files, extracted_contents, knowledge_files = await process_uploaded_files([sample_pdf_file])

        # Assert
        # Should not create direct files anymore as LLMs can't process files directly
        assert len(direct_files) == 0
        assert len(extracted_contents) == 1
        assert len(knowledge_files) == 0

        # Check that extracted content contains PDF file info or actual content
        assert "3. SpringAI.pdf" in extracted_contents[0]

    @pytest.mark.asyncio
    async def test_process_uploaded_files_with_pdf_and_knowledge_base(self, sample_pdf_file):
        """Test processing a PDF file with knowledge base"""
        # Arrange
        mock_knowledge = AsyncMock()
        mock_knowledge.async_load_documents = AsyncMock()

        with patch("agno.file.file_processor._load_file_to_knowledge_base", return_value=True):
            # Act
            direct_files, extracted_contents, knowledge_files = await process_uploaded_files(
                [sample_pdf_file], mock_knowledge
            )

            # Assert
            assert len(direct_files) == 0  # With knowledge base, should not create direct files for large files
            assert len(extracted_contents) == 1  # But should still extract content for small files
            assert len(knowledge_files) == 0  # Knowledge files list is not populated in current implementation

    @pytest.mark.asyncio
    async def test_process_uploaded_files_empty_list(self):
        """Test processing an empty list of files"""
        # Act
        direct_files, extracted_contents, knowledge_files = await process_uploaded_files([])

        # Assert
        assert len(direct_files) == 0
        assert len(extracted_contents) == 0
        assert len(knowledge_files) == 0

    @pytest.mark.asyncio
    async def test_process_uploaded_files_none_files(self):
        """Test processing None as files list"""
        # Act
        direct_files, extracted_contents, knowledge_files = await process_uploaded_files(None)

        # Assert
        assert len(direct_files) == 0
        assert len(extracted_contents) == 0
        assert len(knowledge_files) == 0

    @pytest.mark.asyncio
    async def test_process_uploaded_files_text_file(self):
        """Test processing a text file"""
        # Arrange
        file_content = b"Hello, this is a test text file."
        file = Mock(spec=UploadFile)
        file.filename = "test.txt"
        file.content_type = "text/plain"
        file.read = AsyncMock(return_value=file_content)

        # Act
        direct_files, extracted_contents, knowledge_files = await process_uploaded_files([file])

        # Assert
        # Should not create direct files anymore as LLMs can't process files directly
        assert len(direct_files) == 0
        assert len(extracted_contents) == 1
        assert len(knowledge_files) == 0

        # Check that extracted content contains the text
        assert "Hello, this is a test text file." in extracted_contents[0]

    @pytest.mark.asyncio
    async def test_process_uploaded_files_large_file(self):
        """Test processing a large file (>10MB)"""
        # Arrange
        # Create a large content (15MB)
        file_content = b"A" * (15 * 1024 * 1024)  # 15MB of data
        file = Mock(spec=UploadFile)
        file.filename = "large_file.txt"
        file.content_type = "text/plain"  # Use an allowed MIME type
        file.read = AsyncMock(return_value=file_content)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await process_uploaded_files([file])

        # Check that the exception has the correct status code and detail
        assert exc_info.value.status_code == 400
        assert "too large" in str(exc_info.value.detail)
        assert "large_file.txt" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_process_uploaded_files_medium_file(self):
        """Test processing a medium file (between 1MB and 10MB)"""
        # Arrange
        # Create a medium content (5MB)
        file_content = b"A" * (5 * 1024 * 1024)  # 5MB of data
        file = Mock(spec=UploadFile)
        file.filename = "medium_file.txt"
        file.content_type = "text/plain"
        file.read = AsyncMock(return_value=file_content)

        # Act
        direct_files, extracted_contents, knowledge_files = await process_uploaded_files([file])

        # Assert
        # Should not create direct files anymore
        assert len(direct_files) == 0
        # Should not extract content for files over 1MB when no knowledge base
        assert len(extracted_contents) == 0
        assert len(knowledge_files) == 0

    @pytest.mark.asyncio
    async def test_process_uploaded_files_medium_file_with_knowledge_base(self):
        """Test processing a medium file (between 1MB and 10MB) with knowledge base"""
        # Arrange
        # Create a medium content (5MB)
        file_content = b"A" * (5 * 1024 * 1024)  # 5MB of data
        file = Mock(spec=UploadFile)
        file.filename = "medium_file.txt"
        file.content_type = "text/plain"
        file.read = AsyncMock(return_value=file_content)

        mock_knowledge = AsyncMock()
        mock_knowledge.async_load_documents = AsyncMock()

        # Act
        with patch("agno.file.file_processor._load_file_to_knowledge_base", return_value=True):
            direct_files, extracted_contents, knowledge_files = await process_uploaded_files([file], mock_knowledge)

            # Assert
            # Should not create direct files anymore
            assert len(direct_files) == 0
            # Should not extract content for files over 1MB
            assert len(extracted_contents) == 0
            assert len(knowledge_files) == 0

    @pytest.mark.asyncio
    async def test_process_uploaded_files_small_file_with_knowledge_base(self):
        """Test processing a small file (<1MB) with knowledge base"""
        # Arrange
        file_content = b"Hello, this is a test text file."
        file = Mock(spec=UploadFile)
        file.filename = "test.txt"
        file.content_type = "text/plain"
        file.read = AsyncMock(return_value=file_content)

        mock_knowledge = AsyncMock()
        mock_knowledge.async_load_documents = AsyncMock()

        # Act
        with patch("agno.file.file_processor._load_file_to_knowledge_base", return_value=True):
            direct_files, extracted_contents, knowledge_files = await process_uploaded_files([file], mock_knowledge)

            # Assert
            # Should not create direct files anymore
            assert len(direct_files) == 0
            # Should extract content for files under 1MB
            assert len(extracted_contents) == 1
            assert len(knowledge_files) == 0

    @pytest.mark.asyncio
    async def test_process_uploaded_files_file_without_filename(self):
        """Test processing a file without filename"""
        # Arrange
        file = Mock(spec=UploadFile)
        file.filename = None  # No filename
        file.content_type = "text/plain"
        file.read = AsyncMock(return_value=b"content")

        # Act
        direct_files, extracted_contents, knowledge_files = await process_uploaded_files([file])

        # Assert
        assert len(direct_files) == 0
        assert len(extracted_contents) == 0
        assert len(knowledge_files) == 0

    @pytest.mark.asyncio
    async def test_process_uploaded_files_with_file_error(self):
        """Test processing a file that raises an exception"""
        # Arrange
        file = Mock(spec=UploadFile)
        file.filename = "error_file.txt"
        file.content_type = "text/plain"
        file.read = AsyncMock(side_effect=Exception("Read error"))

        # Act
        direct_files, extracted_contents, knowledge_files = await process_uploaded_files([file])

        # Assert
        assert len(direct_files) == 0
        assert len(extracted_contents) == 0
        assert len(knowledge_files) == 0

    @pytest.mark.asyncio
    async def test_process_uploaded_files_pdf_file_without_actual_file(self):
        """Test processing a PDF file when the actual file doesn't exist"""
        # This test will run when the PDF file doesn't exist
        if os.path.exists("/Users/jiangjiangdear/Documents/3. SpringAI.pdf"):
            pytest.skip("Actual PDF file exists, skipping this test")

        # Arrange
        file_content = b"%PDF-1.4\n%EOF"  # Minimal PDF content
        file = Mock(spec=UploadFile)
        file.filename = "test.pdf"
        file.content_type = "application/pdf"
        file.read = AsyncMock(return_value=file_content)

        # Act
        direct_files, extracted_contents, knowledge_files = await process_uploaded_files([file])

        # Assert
        # Should not create direct files anymore as LLMs can't process files directly
        assert len(direct_files) == 0
        assert len(extracted_contents) == 1  # Should extract content
        assert len(knowledge_files) == 0

        # Check that extracted content contains PDF file info
        assert "test.pdf" in extracted_contents[0]


class TestExtractFileContent:
    """Test cases for _extract_file_content function"""

    @pytest.mark.asyncio
    async def test_extract_pdf_content(self):
        """Test extracting content from a PDF file"""
        # Arrange
        pdf_content = b"%PDF-1.4\n%EOF"  # Minimal PDF content
        file = Mock(spec=UploadFile)
        file.filename = "test.pdf"
        file.content_type = "application/pdf"

        # Mock the PDFReader to return a document with content
        with patch("agno.document.reader.pdf_reader.PDFReader") as mock_pdf_reader:
            mock_instance = mock_pdf_reader.return_value
            mock_doc = Mock()
            mock_doc.content = "This is test PDF content"
            mock_instance.read.return_value = [mock_doc]

            # Act
            result = await _extract_file_content(file, pdf_content)

            # Assert
            assert result is not None
            assert "test.pdf" in result
            # Should contain actual content, not just file info
            assert "This is test PDF content" in result

    @pytest.mark.asyncio
    async def test_extract_docx_content(self):
        """Test extracting content from a DOCX file"""
        # Arrange
        docx_content = b"PK\x03\x04"  # DOCX files are ZIP archives
        file = Mock(spec=UploadFile)
        file.filename = "test.docx"
        file.content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        # Mock the DocxReader to return a document with content
        with patch("agno.document.reader.docx_reader.DocxReader") as mock_docx_reader:
            mock_instance = mock_docx_reader.return_value
            mock_doc = Mock()
            mock_doc.content = "This is test DOCX content"
            mock_instance.read.return_value = [mock_doc]

            # Act
            result = await _extract_file_content(file, docx_content)

            # Assert
            assert result is not None
            assert "test.docx" in result
            # Should contain actual content, not just file info
            assert "This is test DOCX content" in result

    @pytest.mark.asyncio
    async def test_extract_text_content(self):
        """Test extracting content from a text file"""
        # Arrange
        text_content = b"This is test text content."
        file = Mock(spec=UploadFile)
        file.filename = "test.txt"
        file.content_type = "text/plain"

        # Mock the TextReader to return a document with content
        with patch("agno.document.reader.text_reader.TextReader") as mock_text_reader:
            mock_instance = mock_text_reader.return_value
            mock_doc = Mock()
            mock_doc.content = "This is test text content."
            mock_instance.read.return_value = [mock_doc]

            # Act
            result = await _extract_file_content(file, text_content)

            # Assert
            assert result is not None
            assert "This is test text content." in result

    @pytest.mark.asyncio
    async def test_extract_json_content(self):
        """Test extracting content from a JSON file"""
        # Arrange
        json_content = b'{"key": "value"}'
        file = Mock(spec=UploadFile)
        file.filename = "test.json"
        file.content_type = "application/json"

        # Mock the JSONReader to return a document with content
        with patch("agno.document.reader.json_reader.JSONReader") as mock_json_reader:
            mock_instance = mock_json_reader.return_value
            mock_doc = Mock()
            mock_doc.content = '{"key": "value"}'
            mock_instance.read.return_value = [mock_doc]

            # Act
            result = await _extract_file_content(file, json_content)

            # Assert
            assert result is not None
            assert "test.json" in result
            assert '{"key": "value"}' in result

    @pytest.mark.asyncio
    async def test_extract_csv_content(self):
        """Test extracting content from a CSV file"""
        # Arrange
        csv_content = b"name,age\nJohn,25\nJane,30"
        file = Mock(spec=UploadFile)
        file.filename = "test.csv"
        file.content_type = "text/csv"

        # Mock the CSVReader to return a document with content
        with patch("agno.document.reader.csv_reader.CSVReader") as mock_csv_reader:
            mock_instance = mock_csv_reader.return_value
            mock_doc = Mock()
            mock_doc.content = "name, age\nJohn, 25\nJane, 30"
            mock_instance.read.return_value = [mock_doc]

            # Act
            result = await _extract_file_content(file, csv_content)

            # Assert
            assert result is not None
            assert "test.csv" in result
            assert "name, age" in result

    @pytest.mark.asyncio
    async def test_extract_pdf_content_import_error(self):
        """Test extracting content from a PDF file when PDFReader is not available"""
        # Arrange
        pdf_content = b"%PDF-1.4\n%EOF"
        file = Mock(spec=UploadFile)
        file.filename = "test.pdf"
        file.content_type = "application/pdf"

        # Mock ImportError when importing PDFReader
        with patch("agno.document.reader.pdf_reader.PDFReader", side_effect=ImportError):
            # Act
            result = await _extract_file_content(file, pdf_content)

            # Assert
            assert result is not None
            assert "[PDF文件: test.pdf" in result
            assert "大小:" in result

    @pytest.mark.asyncio
    async def test_extract_docx_content_import_error(self):
        """Test extracting content from a DOCX file when DocxReader is not available"""
        # Arrange
        docx_content = b"PK\x03\x04"
        file = Mock(spec=UploadFile)
        file.filename = "test.docx"
        file.content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        # Mock ImportError when importing DocxReader
        with patch("agno.document.reader.docx_reader.DocxReader", side_effect=ImportError):
            # Act
            result = await _extract_file_content(file, docx_content)

            # Assert
            assert result is not None
            assert "[DOCX文件: test.docx" in result
            assert "大小:" in result

    @pytest.mark.asyncio
    async def test_extract_text_content_import_error(self):
        """Test extracting content from a text file when TextReader is not available"""
        # Arrange
        text_content = b"This is test text content."
        file = Mock(spec=UploadFile)
        file.filename = "test.txt"
        file.content_type = "text/plain"

        # Mock ImportError when importing TextReader
        with patch("agno.document.reader.text_reader.TextReader", side_effect=ImportError):
            # Act
            result = await _extract_file_content(file, text_content)

            # Assert
            assert result == "This is test text content."

    @pytest.mark.asyncio
    async def test_extract_json_content_import_error(self):
        """Test extracting content from a JSON file when JSONReader is not available"""
        # Arrange
        json_content = b'{"key": "value"}'
        file = Mock(spec=UploadFile)
        file.filename = "test.json"
        file.content_type = "application/json"

        # Mock ImportError when importing JSONReader
        with patch("agno.document.reader.json_reader.JSONReader", side_effect=ImportError):
            # Act
            result = await _extract_file_content(file, json_content)

            # Assert
            assert result == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_extract_csv_content_import_error(self):
        """Test extracting content from a CSV file when CSVReader is not available"""
        # Arrange
        csv_content = b"name,age\nJohn,25\nJane,30"
        file = Mock(spec=UploadFile)
        file.filename = "test.csv"
        file.content_type = "text/csv"

        # Mock ImportError when importing CSVReader
        with patch("agno.document.reader.csv_reader.CSVReader", side_effect=ImportError):
            # Act
            result = await _extract_file_content(file, csv_content)

            # Assert
            assert result == "name,age\nJohn,25\nJane,30"
