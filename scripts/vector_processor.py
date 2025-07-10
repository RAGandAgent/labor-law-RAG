import logging
import os
import time
from http import HTTPStatus

import dashscope
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, RRFRanker, AnnSearchRequest

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from utils import json_to_chunks, txt_to_chunks
    from document_parser import ParsedRecordManager
except:
    from utils import json_to_chunks, txt_to_chunks
    from document_parser import ParsedRecordManager


class VectorProcessor:

    def __init__(self, milvus_host="127.0.0.1", milvus_port="19530",
                 dashscope_api_key="", drop_collection=[],
                 record_manager: ParsedRecordManager = None):
        self.milvus_client = MilvusClient(host=milvus_host, port=milvus_port)
        self.record_manager = record_manager
        dashscope.api_key = dashscope_api_key

        for dcoll in drop_collection:
            self.milvus_client.drop_collection(collection_name=dcoll)
            logging.info(f"Collection '{dcoll}' dropped.")
            time.sleep(1)

        collections = self.milvus_client.list_collections()
        logging.info(f"当前Milvus中的所有Collection: {collections}")
        logging.info(f"向量数据库启动成功")

    def _create_collection(self, collection_name):
        if self.milvus_client.has_collection(collection_name=collection_name):
            logging.info(f"Collection '{collection_name}' already exists.")
            # 如果集合已存在，也需要加载到内存
            self.milvus_client.load_collection(collection_name=collection_name)
            logging.info(f"Collection '{collection_name}' loaded into memory.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="text_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256, description="原始文件名"),
            FieldSchema(name="page_number", dtype=DataType.VARCHAR, max_length=128, description="文本所在页码")
        ]
        schema = CollectionSchema(fields, description=f"{collection_name} RAG Collection")
        self.milvus_client.create_collection(collection_name=collection_name, schema=schema)
        logging.info(f"Collection '{collection_name}' created successfully.")

        # 创建索引
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="L2")
        index_params.add_index(field_name="text_sparse", index_type="SPARSE_INVERTED_INDEX",
                               metric_type="IP",
                               params={"inverted_index_algo": "DAAT_MAXSCORE"})
        self.milvus_client.create_index(collection_name=collection_name, index_params=index_params)
        logging.info(f"Index created for collection '{collection_name}'.")

        # 加载集合到内存
        self.milvus_client.load_collection(collection_name=collection_name)
        logging.info(f"Collection '{collection_name}' loaded into memory.")

    def emb_text(self, text, is_query=False):
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=text,
            dimension=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            output_type="dense&sparse"
        )
        if resp.status_code == HTTPStatus.OK:
            dense_embedding = resp.output['embeddings'][0]['embedding']
            sparse_embedding_data = resp.output['embeddings'][0]['sparse_embedding']
            sparse_embedding_dict = {}
            for item in sparse_embedding_data:
                sparse_embedding_dict[item['index']] = item['value']
            return dense_embedding, sparse_embedding_dict
        else:
            logging.error(
                f"Error getting embedding for text: {text}. Status code: {resp.status_code}, Message: {resp.message}")
            return None, None

    def save_chunks(self, chunks, file_name, collection_name):
        entities = []
        for chunk_item in chunks:
            chunk_text = chunk_item['text']
            if chunk_text:
                page_number = chunk_item['page_number']
                dense_embedding, sparse_embedding = self.emb_text(chunk_text)  # Get both dense and sparse embeddings
                if dense_embedding is None or sparse_embedding is None:
                    logging.warning(f"Skipping chunk due to embedding failure: {chunk_text[:50]}...")
                    continue
                entity_data = {
                    "embedding": dense_embedding,
                    "text": chunk_text,
                    "file_name": file_name,
                    "page_number": ','.join(map(str, page_number)),
                    "text_sparse": sparse_embedding
                }
                entities.append(entity_data)

        if entities:
            logging.debug(f"Attempting to insert {len(entities)} entities into {collection_name}")
            res = self.milvus_client.insert(collection_name=collection_name, data=entities)
            logging.info(f"Inserted {len(res['ids'])} documents into Milvus collection {collection_name}.")
            return res['ids']
        return []

    def vectorize_parsed_documents(self, chunk_size: int = 500, chunk_overlap_percent: float = 0.1) -> list:
        results = []
        chunk_overlap = int(chunk_size * chunk_overlap_percent)

        for record in self.record_manager.records:
            ori_filename = os.path.basename(record['original_filename'])
            parsed_filename = record['filename']
            coll_name = record['collection']
            if self.record_manager.record_status_is_embed(record):
                logging.info(f"文件 {parsed_filename} 已经向量化，已忽略")
                continue

            self._create_collection(coll_name)

            result = dict()
            result['name'] = ori_filename
            result['inserted_ids'] = []

            logging.info(f"数据集: {coll_name}, 文件: {parsed_filename}, 原文件: {ori_filename} 正在向量化......")
            file_content = self.record_manager.read_document(parsed_filename)

            if not file_content:
                logging.warning(f"读取内容为空: {parsed_filename}")
                continue

            chunks = []
            if parsed_filename.endswith('.json'):
                chunks = json_to_chunks(file_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap_percent)
            elif parsed_filename.endswith('.txt'):
                chunks = txt_to_chunks(
                    text_content=file_content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap_percent,
                    separators=["\n\n", "#"]
                )
            else:
                logging.warning(f"不支持的文件类型: {parsed_filename}")
                continue

            inserted_ids = self.save_chunks(chunks, ori_filename, coll_name)
            result['inserted_ids'].extend(inserted_ids)

            result['size'] = len(result['inserted_ids'])
            logging.info(f"数据集: {coll_name}, 文件 '{parsed_filename}' 处理完成，共插入 {result['size']} 个文本块。")
            self.record_manager.record_update_status_embed(record)
            results.append(result)
        return results

    def search_hybrid(self, collection_name, query, count=100, top_k=5):
        t0 = time.time()
        dense_embedding, sparse_embedding = self.emb_text(query)
        reqs = []

        if dense_embedding is not None:
            query_embedding_params = {
                "data": [dense_embedding],
                "anns_field": "embedding",
                "param": {"nprobe": 10},
                "limit": count
            }
            query_emb_req = AnnSearchRequest(**query_embedding_params)
            reqs.append(query_emb_req)
        if sparse_embedding is not None:
            query_text_params = {
                "data": [sparse_embedding],
                "anns_field": "text_sparse",
                "param": {"nprobe": 10},
                "limit": count
            }
            query_text_req = AnnSearchRequest(**query_text_params)
            reqs.append(query_text_req)
        else:
            logging.warning(f"Warning: No dense embedding generated for query: '{query}'. Cannot perform search.")

        if not reqs:
            logging.error(f"Error: No valid dense embedding generated for query: '{query}'. Cannot perform search.")
            return []

        ranker = RRFRanker(count)
        search_result = self.milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=top_k,
            output_fields=["id", "text", "file_name", "page_number"]
        )
        t1 = time.time()
        logging.info(
            f"检索到的文档: {len(search_result[0])} 个，耗时: {t1 - t0:.2f} 秒，最高分：{search_result[0][0]['distance']}，最低分：{search_result[0][-1]['distance']}")
        results = []
        for item in search_result[0]:
            entity = item['entity']
            entity['score'] = item['distance']
            results.append(entity)
        return results


if __name__ == '__main__':
    vector = VectorProcessor(dashscope_api_key='YOUR_KEY')
    vector.vectorize_parsed_documents()

