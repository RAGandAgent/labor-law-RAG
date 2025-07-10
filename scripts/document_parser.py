import io
import json
import logging
import os
import time
import zipfile
from typing import Union

import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ParsedRecordManager:
    def __init__(self, output_dir, record_filename):
        self.output_dir = output_dir
        self.record_file_path = os.path.join(output_dir, record_filename)
        self.records = self._load_records()

    def _load_records(self):
        if os.path.exists(self.record_file_path):
            try:
                with open(self.record_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {self.record_file_path}. Starting with empty records.")
                return []
        return []

    def read_document(self, file_name: str)->Union[list, str, dict]:
        filepath = os.path.join(self.output_dir, file_name)
        if not os.path.exists(filepath):
            logging.error(f"Error: File not found at {filepath}")
            return []

        try:
            if file_name.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_name.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logging.warning(f"Unsupported file type for reading: {file_name}")
                return []
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {filepath}: {e}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred while reading {filepath}: {e}")
            return []

    def save_records(self):
        os.makedirs(os.path.dirname(self.record_file_path), exist_ok=True)
        with open(self.record_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=4, ensure_ascii=False)
        logging.info(f"Saved parsed records to {self.record_file_path}")

    def add_record(self, record):
        if self.find_record_idx(record) != 0:
            self.records.append(record)

    def has_record(self, pdf_file: str) -> bool:
        return any(record.get('original_filename') == pdf_file for record in self.records)

    def find_record_idx(self, record: dict):
        for idx, r in enumerate(self.records):
            if r.get('filename') == record.get('filename'):
                return idx
        return None

    def record_status_is_embed(self, record: dict):
        idx = self.find_record_idx(record)
        return self.records[idx].get('status') == 'embed'

    def record_update_status_embed(self, record: dict):
        idx = self.find_record_idx(record)
        self.records[idx]['status'] = 'embed'
        self.save_records()


class MineruParser:

    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def upload_files_batch(self, file_paths, enable_formula=True, language="ch", enable_table=True):
        url = 'https://mineru.net/api/v4/file-urls/batch'
        files_data = []
        for fp in file_paths:
            file_name = os.path.basename(fp)
            files_data.append({"name": file_name, "is_ocr": True, "data_id": file_name})

        data = {
            "enable_formula": enable_formula,
            "language": language,
            "enable_table": enable_table,
            "files": files_data
        }

        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Mineru API:{url} Response: {result}")
                if result["code"] == 0:
                    batch_id = result["data"]["batch_id"]
                    urls = result["data"]["file_urls"]
                    logging.info(f'Mineru: Batch ID: {batch_id}, Upload URLs obtained.')
                    # 上传文件到Mineru提供的URL
                    for i, upload_url in enumerate(urls):
                        with open(file_paths[i], 'rb') as f:
                            res_upload = requests.put(upload_url, data=f)
                            if res_upload.status_code == 200:
                                logging.info(f"Mineru: {os.path.basename(file_paths[i])} uploaded successfully.")
                            else:
                                logging.error(
                                    f"Mineru: {os.path.basename(file_paths[i])} upload failed: {res_upload.status_code} {res_upload.text}")
                    return batch_id, urls
                else:
                    logging.error(f'Mineru: Failed to get upload URLs: {result.get("msg", "Unknown error")}')
            else:
                logging.error(
                    f'Mineru: API response not successful. Status: {response.status_code}, Result: {response.text}')
        except Exception as err:
            logging.error(f'Mineru: Error during file upload batch: {err}')
        return None, None

    def get_extract_results_batch(self, batch_id, timeout=60, interval=5, output_dir=""):
        url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
        start_time = time.time()
        last_response = None
        fina_results = []
        while time.time() - start_time < timeout:
            try:
                res = requests.get(url, headers=self.headers)
                last_response = res
                if res.status_code == 200:
                    result = res.json()
                    logging.info(f"Mineru API({url}) Response: {result}")
                    if result["code"] == 0:
                        extract_results = result["data"].get("extract_result", [])
                        logging.info(f"Mineru extract_results: {extract_results}")
                        # 检查是否有文件处于running状态
                        if any(item.get("state") != "done" for item in extract_results):
                            logging.info(f"Mineru: Batch {batch_id} still processing... Waiting {interval} seconds.")
                        else:
                            # 所有文件都已处理完毕（done或failed）
                            logging.info(f"Mineru: Batch {batch_id} results obtained successfully.")
                            for item in extract_results:
                                if item.get("state") == "done" and item.get("full_zip_url"):
                                    full_zip_url = item["full_zip_url"]
                                    jsondata_file = self._process_zip_file(full_zip_url, output_dir=output_dir)
                                    if jsondata_file:
                                        fina_results.append((item['file_name'], jsondata_file))
                            return fina_results
                    else:
                        logging.error(f'Mineru: Failed to get extract results: {result.get("msg", "Unknown error")}')
                else:
                    logging.error(f'Mineru: API response not successful. Status: {res.status_code}, Result: {res.text}')
            except Exception as err:
                logging.error(f'Mineru: Error during getting extract results: {err}')
            time.sleep(interval)
        logging.warning(f"Mineru: Timeout waiting for batch {batch_id} results.")
        return fina_results

    def _process_zip_file(self, full_zip_url, output_dir):
        try:
            zip_response = requests.get(full_zip_url)
            zip_response.raise_for_status()
            logging.info(f"Download ZIP Response for {full_zip_url}: {zip_response.status_code}")
            with zipfile.ZipFile(io.BytesIO(zip_response.content), 'r') as zf:
                for zf_info in zf.infolist():
                    # 查找JSON文件
                    if zf_info.filename != "layout.json" and zf_info.filename.endswith('.json'):
                        with zf.open(zf_info.filename) as json_file:
                            json_content = json_file.read().decode('utf-8')
                            base_name_without_ext = os.path.splitext(os.path.basename(full_zip_url))[0] + '.json'
                            output_json_path = os.path.join(output_dir, base_name_without_ext)
                            os.makedirs(output_dir, exist_ok=True)
                            with open(output_json_path, "w", encoding="utf-8") as f:
                                f.write(json_content)
                            logging.info(f"Saved extracted JSON to {base_name_without_ext}")
                            return base_name_without_ext
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download zip from {full_zip_url}: {e}")
            return None
        except zipfile.BadZipFile:
            logging.error(f"Downloaded file is not a valid zip file from {full_zip_url}")
            return None
        except Exception as e:
            logging.error(f"An error occurred during zip processing: {e}")
            return None


if __name__ == '__main__':
    mineru_api_key = "YOUR_KEY"
    files_path = ["中人民共和国劳动法.pdf"]
    out_dir = "./dd"
    parser = MineruParser(mineru_api_key)
    batch_id, _ = parser.upload_files_batch(files_path)
    if batch_id:
        file_results = parser.get_extract_results_batch(batch_id, output_dir=out_dir)
        print(f"get_extract_results_batch: {file_results}")
        if file_results:
            for src_file, json_file in file_results:
                record = {
                    "filename": json_file,
                    "original_filename": src_file,
                    "collection": "pa_baoxian"
                }
                print("record:", record)
