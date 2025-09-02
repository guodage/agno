from io import BytesIO
from typing import List, Optional, Tuple
from fastapi import UploadFile, HTTPException
from agno.media import File as FileMedia
from agno.utils.log import logger
from agno.document import Document


async def _extract_content_with_reader(reader_class: type, file_content: bytes, filename: str) -> Optional[str]:
    """
    使用指定的读取器类提取文件内容

    Args:
        reader_class: 读取器类（如PDFReader, DocxReader等）
        file_content: 文件内容字节
        filename: 文件名

    Returns:
        提取的文本内容或None
    """
    try:
        file_io = BytesIO(file_content)
        file_io.name = filename or "file"
        documents = reader_class().read(file_io)
        # 将所有文档内容连接起来
        content = "\n\n".join([doc.content for doc in documents if doc.content])
        return content if content else None
    except ImportError:
        # 重新抛出ImportError，以便调用者可以正确处理
        raise
    except Exception as e:
        logger.error(f"Error extracting content with {reader_class.__name__}: {e}")
        return None


async def _load_content_to_knowledge_base(reader_class: type, file_content: bytes, filename: str,
                                        agent_knowledge, file_type: str) -> Tuple[bool, Optional[List[Document]]]:
    """
    使用指定的读取器类将文件内容加载到知识库

    Args:
        reader_class: 读取器类（如PDFReader, DocxReader等）
        file_content: 文件内容字节
        filename: 文件名
        agent_knowledge: Agent的知识库
        file_type: 文件类型描述

    Returns:
        (是否成功加载, 提取的Document列表)
    """
    try:
        file_io = BytesIO(file_content)
        file_io.name = filename
        file_content_docs = reader_class().read(file_io)
        await agent_knowledge.async_load_documents(file_content_docs, upsert=True)
        logger.info(f"Loaded {file_type} document '{filename}' into agent's knowledge base")
        return True, file_content_docs
    except ImportError:
        logger.warning(f"{reader_class.__name__} not available, skipping {file_type} processing")
        return False, None
    except Exception as e:
        logger.error(f"Error loading {file_type} {filename} to knowledge base: {e}")
        return False, None


async def process_uploaded_files(
    files: Optional[List[UploadFile]],
    agent_knowledge=None
) -> Tuple[List[FileMedia], List[str], List[FileMedia]]:
    """
    统一处理上传的文件，支持两种模式：
    1. 小文件进行文本解析并直接传递给Agent
    2. 大文件或有知识库的Agent将文件内容加载到知识库中

    Args:
        files: 上传的文件列表
        agent_knowledge: Agent的知识库（可选）

    Returns:
        Tuple of (direct_files, extracted_contents, knowledge_files)
    """
    direct_files: List[FileMedia] = []
    extracted_contents: List[str] = []
    knowledge_files: List[FileMedia] = []

    if not files:
        return direct_files, extracted_contents, knowledge_files

    for upload_file in files:
        if not upload_file.filename:
            continue

        try:
            content = await upload_file.read()
            file_size = len(content)

            if file_size > 1024 * 1024 * 100:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {upload_file.filename} is too large ({file_size} bytes) to process. "
                           f"Maximum allowed size is 100MB."
                )

            extracted_text = None

            if agent_knowledge is not None:
                success, text_from_kb = await _load_file_to_knowledge_base(
                    upload_file, content, agent_knowledge
                )
                if success:
                    extracted_text = text_from_kb

            if extracted_text is None:
                extracted_text = await _extract_file_content(upload_file, content)

            if file_size <= 1024 * 1024 * 5 and extracted_text:
                extracted_contents.append(f"Content of {upload_file.filename}:\n{extracted_text}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing file {upload_file.filename}: {e}")
            continue

    return direct_files, extracted_contents, knowledge_files


async def _extract_file_content(upload_file: UploadFile, content: bytes) -> Optional[str]:
    try:
        if upload_file.content_type == "text/plain":
            try:
                from agno.document.reader.text_reader import TextReader
                extracted_content = await _extract_content_with_reader(TextReader, content, upload_file.filename)
                if extracted_content:
                    return extracted_content
            except ImportError:
                logger.warning("Text reader not available, using simple decoding")
            # Fallback to simple decoding
            return content.decode('utf-8', errors='ignore')
        elif upload_file.content_type == "application/json":
            try:
                from agno.document.reader.json_reader import JSONReader
                extracted_content = await _extract_content_with_reader(JSONReader, content, upload_file.filename)
                if extracted_content:
                    return extracted_content
            except ImportError:
                logger.warning("JSON reader not available, using simple decoding")
            # Fallback to simple decoding
            return content.decode('utf-8', errors='ignore')
        elif upload_file.content_type == "text/csv":
            try:
                from agno.document.reader.csv_reader import CSVReader
                extracted_content = await _extract_content_with_reader(CSVReader, content, upload_file.filename)
                if extracted_content:
                    return extracted_content
            except ImportError:
                logger.warning("CSV reader not available, using simple decoding")
            # Fallback to simple decoding
            return content.decode('utf-8', errors='ignore')
        elif upload_file.content_type == "application/pdf":
            try:
                from agno.document.reader.pdf_reader import PDFReader
                extracted_content = await _extract_content_with_reader(PDFReader, content, upload_file.filename)
                if extracted_content:
                    return extracted_content
            except ImportError:
                logger.warning("pypdf not installed, skipping PDF content extraction")
            except Exception as e:
                logger.error(f"Error extracting content from PDF {upload_file.filename}: {e}")
            # Fallback to file info
            return f"[PDF文件: {upload_file.filename}, 大小: {len(content)} bytes]"
        elif upload_file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                from agno.document.reader.docx_reader import DocxReader
                extracted_content = await _extract_content_with_reader(DocxReader, content, upload_file.filename)
                if extracted_content:
                    return extracted_content
            except ImportError:
                logger.warning("python-docx not installed, skipping DOCX content extraction")
            except Exception as e:
                logger.error(f"Error extracting content from DOCX {upload_file.filename}: {e}")
            # Fallback to file info
            return f"[DOCX文件: {upload_file.filename}, 大小: {len(content)} bytes]"

    except Exception as e:
        logger.error(f"Error extracting content from {upload_file.filename}: {e}")

    return None


async def _load_file_to_knowledge_base(upload_file: UploadFile, content: bytes, agent_knowledge) -> Tuple[bool, Optional[str]]:
    """将文件加载到知识库中, 并返回提取的文本"""
    content_type = upload_file.content_type
    reader_map = {
        "application/pdf": ("agno.document.reader.pdf_reader.PDFReader", "PDF"),
        "text/csv": ("agno.document.reader.csv_reader.CSVReader", "CSV"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ("agno.document.reader.docx_reader.DocxReader", "DOCX"),
        "text/plain": ("agno.document.reader.text_reader.TextReader", "text"),
        "application/json": ("agno.document.reader.json_reader.JSONReader", "JSON"),
    }

    if content_type not in reader_map:
        return False, None

    reader_path, file_type = reader_map[content_type]

    try:
        parts = reader_path.split('.')
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        module = __import__(module_path, fromlist=[class_name])
        reader_class = getattr(module, class_name)

        success, documents = await _load_content_to_knowledge_base(
            reader_class, content, upload_file.filename, agent_knowledge, file_type)

        if success and documents:
            extracted_text = "\n\n".join([doc.content for doc in documents if doc.content])
            return True, extracted_text

        return success, None

    except ImportError:
        logger.warning(f"Reader for {content_type} not available, skipping processing")
        return False, None
