import logging
import os
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from document_parser import ParsedRecordManager, MineruParser
from vector_processor import VectorProcessor
from utils import get_dir_and_file_names, generate_uuid


class Pipeline:
    def __init__(self, mineru_api_key='', dashscope_api_key='', parsed_output_dir='',
                 record_filename='parsed_records.json') -> None:
        self.parsed_output_dir = parsed_output_dir
        self.record_manager = ParsedRecordManager(output_dir=self.parsed_output_dir, record_filename=record_filename)
        self.mineru_parser = MineruParser(mineru_api_key)
        self.vector = VectorProcessor(dashscope_api_key=dashscope_api_key, record_manager=self.record_manager)
        logging.info("pipeline初始化成功")

    def _parse_single_document(self, file_path, collection_name):
        file_extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        if self.record_manager and self.record_manager.has_record(file_name):
            logging.info(f"Skipping {file_path} as it's already parsed.")
            return False

        if file_extension == '.txt':
            new_file_name = f"{generate_uuid()}.txt"
            destination_path = os.path.join(self.parsed_output_dir, new_file_name)
            try:
                os.makedirs(self.parsed_output_dir, exist_ok=True)
                shutil.copy(file_path, destination_path)
                logging.info(f"Moved {file_path} to {destination_path}")
                record = {
                    "filename": new_file_name,
                    "original_filename": file_name,
                    "collection": collection_name
                }
                self.record_manager.add_record(record)
                self.record_manager.save_records()
                return True
            except Exception as e:
                logging.error(f"Error moving file {file_path}: {e}")
                return False
        elif file_extension == '.pdf':
            logging.info(f"正在上传文件：{file_path}")
            batch_id, _ = self.mineru_parser.upload_files_batch([file_path])
            if batch_id:
                file_results = self.mineru_parser.get_extract_results_batch(batch_id, output_dir=self.parsed_output_dir)
                logging.info(f"get_extract_results_batch: {file_results}")
                if file_results:
                    for src_file, json_file in file_results:
                        record = {
                            "filename": json_file,
                            "original_filename": src_file,
                            "collection": collection_name
                        }
                        self.record_manager.add_record(record)
                    self.record_manager.save_records()
                    return True
            return False
        else:
            logging.warning(f"Unsupported file type for {file_path}. Only .txt and .pdf are supported.")
            return False

    def parse_documents_in_directory(self, path):
        dir_name, files = get_dir_and_file_names(path)
        logging.info(f"待处理目录: {dir_name}, 文件[{len(files)}]: {files}")
        if not files:
            logging.info("No files found to process.")
            return []

        processed_count = 0
        for file_path in files:
            if self._parse_single_document(file_path, dir_name):
                processed_count += 1

        if processed_count > 0:
            logging.info(f"成功处理了 {processed_count} 个文件。")

        return self.record_manager.records if self.record_manager else []

    def parse_documents(self, path):
        if os.path.isfile(path):
            dir_name = os.path.basename(os.path.dirname(path))
            self._parse_single_document(path, dir_name)
            return self.record_manager.records if self.record_manager else []
        elif os.path.isdir(path):
            return self.parse_documents_in_directory(path)
        else:
            logging.error(f"Error: Path {path} is neither a file nor a directory.")
            return []

    def format_search_results(self, data):
        contents = []
        for res in data:
            contents.append("文件：{}(页码：{})\n\n相关度：{}\n\n内容：{}".format(
                res['file_name'], res['page_number'], res['score'], res['text']))
        if len(contents) > 0:
            return "\n\n---\n\n".join(contents)
        return ""

    def vectorize_documents(self):
        return self.vector.vectorize_parsed_documents()

    def search(self, coll, query, **kwargs):
        return self.vector.search_hybrid(coll, query, **kwargs)


output_dir = './parsed_documents'
mineru_api_key = "YOUR_KEY"
dashscope_api_key = "YOUR_KEY"
singleton_pipeline = Pipeline(mineru_api_key=mineru_api_key, dashscope_api_key=dashscope_api_key,
                              parsed_output_dir=output_dir,
                              record_filename="parsed_records.json")
