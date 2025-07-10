import hashlib
import uuid
from typing import List, Dict, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def generate_uuid():
    return str(uuid.uuid4()).replace('-', '')


def generate_md5(input_string: str) -> str:
    return hashlib.md5(input_string.encode('utf-8')).hexdigest()


def get_dir_and_file_names(path):
    file_names = []
    dir_name = ""

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                file_names.append(os.path.join(root, file))
        dir_name = os.path.basename(os.path.normpath(path))
    elif os.path.isfile(path):
        file_names.append(os.path.basename(path))
        dir_name = os.path.basename(os.path.dirname(path))

    return dir_name, file_names


def json_to_chunks(json_content_list: list, chunk_size: int, chunk_overlap: float) -> List[Dict[str, any]]:
    combined_content_with_pages = []
    for item in json_content_list:
        item_type = item.get('type')
        page_number = item.get('page_idx') + 1
        if item_type == 'text':
            text_content = item.get('text')
            if text_content:
                combined_content_with_pages.append({"text": text_content, "page_number": page_number})
        elif item_type == 'table':
            table_body = item.get('table_body')
            if table_body:
                combined_content_with_pages.append({"text": table_body, "page_number": page_number})

    all_chunks_with_pages = []
    current_chunk_text = ""
    current_chunk_pages = set()

    for i, item in enumerate(combined_content_with_pages):
        text_to_add = item['text']
        page_number_to_add = item['page_number']
        if len(current_chunk_text) + len(text_to_add) <= chunk_size:
            current_chunk_text += text_to_add
            current_chunk_pages.add(page_number_to_add)
        else:
            if current_chunk_text:
                all_chunks_with_pages.append({
                    'text': current_chunk_text,
                    'page_number': sorted(list(current_chunk_pages))
                })

            current_chunk_text = ""
            current_chunk_pages = set()

            overlap_start_idx = i - 1
            temp_overlap_text = ""
            temp_overlap_pages = set()

            while overlap_start_idx >= 0 and len(temp_overlap_text) < chunk_overlap:
                prev_item = combined_content_with_pages[overlap_start_idx]
                temp_overlap_text = prev_item['text'] + temp_overlap_text
                temp_overlap_pages.add(prev_item['page_number'])
                overlap_start_idx -= 1

            current_chunk_text = temp_overlap_text + text_to_add
            current_chunk_pages.update(temp_overlap_pages)
            current_chunk_pages.add(page_number_to_add)

    if current_chunk_text:
        all_chunks_with_pages.append({
            'text': current_chunk_text,
            'page_number': sorted(list(current_chunk_pages))
        })

    return all_chunks_with_pages


def txt_to_chunks(text_content: str, chunk_size: int, chunk_overlap: float, separators: list = None) -> List[
    Dict[str, any]]:
    if not separators:
        separators = ["\n\n", "#"]
    else:
        separators = separators

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=separators
    )

    chunk = text_splitter.create_documents([text_content])

    formatted_chunks = []
    for i, chunk in enumerate(chunk):
        formatted_chunks.append({
            'text': chunk.page_content,
            'page_number': [1]
        })
    return formatted_chunks


if __name__ == '__main__':
    with open('demo.txt', 'r', encoding='utf-8') as f:
        text_content = f.read()
        chunks = txt_to_chunks(text_content=text_content, chunk_size=100, chunk_overlap=0.1)
        print(chunks)
