import json
from collections import defaultdict
from typing import Any, Optional

from pydantic import BaseModel

from configs import dify_config
from core.rag.datasource.keyword.jieba.jieba_keyword_table_handler import JiebaKeywordTableHandler
from core.rag.datasource.keyword.keyword_base import BaseKeyword
from core.rag.models.document import Document
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from models.dataset import Dataset, DatasetKeywordTable, DocumentSegment


class KeywordTableConfig(BaseModel):
    max_keywords_per_chunk: int = 10


class Jieba(BaseKeyword):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self._config = KeywordTableConfig()

    def create(self, texts: list[Document], **kwargs) -> BaseKeyword:
        #首先构建一个锁名称，基于数据集的ID
        lock_name = 'keyword_indexing_lock_{}'.format(self.dataset.id)
        # 使用redis_client.lock获取一个锁，确保在关键词索引过程中，同一时间只有一个进程可以执行这个操作。锁的超时时间设置为600秒
        with redis_client.lock(lock_name, timeout=600):
            keyword_table_handler = JiebaKeywordTableHandler()
            keyword_table = self._get_dataset_keyword_table()
            for text in texts:
                #提取的关键词数量由 _config.max_keywords_per_chunk 决定
                keywords = keyword_table_handler.extract_keywords(text.page_content, self._config.max_keywords_per_chunk)
                self._update_segment_keywords(self.dataset.id, text.metadata['doc_id'], list(keywords))
                # 将文本的ID 和 提取的 关键词 添加到关键词表中
                keyword_table = self._add_text_to_keyword_table(keyword_table, text.metadata['doc_id'], list(keywords))

            self._save_dataset_keyword_table(keyword_table)

            return self

    def add_texts(self, texts: list[Document], **kwargs):
        #首先构建一个锁名称，基于数据集的ID
        lock_name = 'keyword_indexing_lock_{}'.format(self.dataset.id)
        # 使用redis_client.lock获取一个锁，确保在关键词索引过程中，同一时间只有一个进程可以执行这个操作。锁的超时时间设置为600秒
        with redis_client.lock(lock_name, timeout=600):
            keyword_table_handler = JiebaKeywordTableHandler()

            keyword_table = self._get_dataset_keyword_table()
            keywords_list = kwargs.get('keywords_list', None)
            for i in range(len(texts)):
                text = texts[i]
                if keywords_list:
                    keywords = keywords_list[i]
                    if not keywords:
                        keywords = keyword_table_handler.extract_keywords(text.page_content,
                                                                          self._config.max_keywords_per_chunk)
                else:
                    keywords = keyword_table_handler.extract_keywords(text.page_content, self._config.max_keywords_per_chunk)
                self._update_segment_keywords(self.dataset.id, text.metadata['doc_id'], list(keywords))
                keyword_table = self._add_text_to_keyword_table(keyword_table, text.metadata['doc_id'], list(keywords))

            self._save_dataset_keyword_table(keyword_table)

    def text_exists(self, id: str) -> bool:
        keyword_table = self._get_dataset_keyword_table()
        return id in set.union(*keyword_table.values())

    def delete_by_ids(self, ids: list[str]) -> None:
        #首先构建一个锁名称，基于数据集的ID
        lock_name = 'keyword_indexing_lock_{}'.format(self.dataset.id)
        # 使用redis_client.lock获取一个锁，确保在关键词索引过程中，同一时间只有一个进程可以执行这个操作。锁的超时时间设置为600秒
        with redis_client.lock(lock_name, timeout=600):
            keyword_table = self._get_dataset_keyword_table()
            keyword_table = self._delete_ids_from_keyword_table(keyword_table, ids)

            self._save_dataset_keyword_table(keyword_table)

    def search(
            self, query: str,
            **kwargs: Any
    ) -> list[Document]:
        # 获取关键词表
        keyword_table = self._get_dataset_keyword_table()

        k = kwargs.get('top_k', 4)
        # 利用 query 生成的 关键词 匹配 关键词表 中数据块并进行排序
        sorted_chunk_indices = self._retrieve_ids_by_query(keyword_table, query, k)
        # 根据检索到的数据块查询数据库获取信息构造 Document 列表
        documents = []
        for chunk_index in sorted_chunk_indices:
            segment = db.session.query(DocumentSegment).filter(
                DocumentSegment.dataset_id == self.dataset.id,
                DocumentSegment.index_node_id == chunk_index
            ).first()

            if segment:

                documents.append(Document(
                    page_content=segment.content,
                    metadata={
                        "doc_id": chunk_index,
                        "doc_hash": segment.index_node_hash,
                        "document_id": segment.document_id,
                        "dataset_id": segment.dataset_id,
                    }
                ))

        return documents

    def delete(self) -> None:
        #首先构建一个锁名称，基于数据集的ID
        lock_name = 'keyword_indexing_lock_{}'.format(self.dataset.id)
        with redis_client.lock(lock_name, timeout=600):
            dataset_keyword_table = self.dataset.dataset_keyword_table
            if dataset_keyword_table:
                db.session.delete(dataset_keyword_table)
                db.session.commit()
                if dataset_keyword_table.data_source_type != 'database':
                    file_key = 'keyword_files/' + self.dataset.tenant_id + '/' + self.dataset.id + '.txt'
                    storage.delete(file_key)

    #将更新后的关键词表保存回数据库或其它存储介质
    def _save_dataset_keyword_table(self, keyword_table):
        keyword_table_dict = {
            '__type__': 'keyword_table',
            '__data__': {
                "index_id": self.dataset.id,
                "summary": None,
                "table": keyword_table
            }
        }
        dataset_keyword_table = self.dataset.dataset_keyword_table
        keyword_data_source_type = dataset_keyword_table.data_source_type
        if keyword_data_source_type == 'database':
            # 序列化 关键词表 为JSON字符串
            dataset_keyword_table.keyword_table = json.dumps(keyword_table_dict, cls=SetEncoder)
            db.session.commit()
        else:
            file_key = 'keyword_files/' + self.dataset.tenant_id + '/' + self.dataset.id + '.txt'
            if storage.exists(file_key):
                storage.delete(file_key)
            storage.save(file_key, json.dumps(keyword_table_dict, cls=SetEncoder).encode('utf-8'))
    # 获取当前数据集的关键词表。如果关键词表不存在，则会根据配置创建一个新的关键词表，并保存到数据库或其它指定的数据源中
    def _get_dataset_keyword_table(self) -> Optional[dict]:
        dataset_keyword_table = self.dataset.dataset_keyword_table             # 获取数据集关键词表
        if dataset_keyword_table:
            keyword_table_dict = dataset_keyword_table.keyword_table_dict
            if keyword_table_dict:
                return keyword_table_dict['__data__']['table']
        else:  # 如果数据集关键词表不存在
            keyword_data_source_type = dify_config.KEYWORD_DATA_SOURCE_TYPE    # 获取关键词数据源类型
            dataset_keyword_table = DatasetKeywordTable(
                dataset_id=self.dataset.id,
                keyword_table='',
                data_source_type=keyword_data_source_type,
            )
            if keyword_data_source_type == 'database':                         # 如果关键词数据源类型是数据库
                dataset_keyword_table.keyword_table = json.dumps({
                    '__type__': 'keyword_table',
                    '__data__': {
                        "index_id": self.dataset.id,
                        "summary": None,
                        "table": {}
                    }
                }, cls=SetEncoder)                                             # 设置关键词表
            db.session.add(dataset_keyword_table)
            db.session.commit()

        return {}
    # 将 一组关键词 与一个特定的 文档段ID 关联起来，并更新关键词表 
    # 参数 keyword_table 关键词表字典 
    # 参数 id 一个文档段的id
    def _add_text_to_keyword_table(self, keyword_table: dict, id: str, keywords: list[str]) -> dict:
        for keyword in keywords:
            if keyword not in keyword_table:
                keyword_table[keyword] = set()    #如果这个关键词在关键词表中不存在，则首先为该关键词创建一个新的ID集合，然后将当前文档段的ID添加到这个新集合中
            keyword_table[keyword].add(id)        #已经存在于关键词表中，则将当前文档段的ID添加到该关键词对应的ID集合中
            #最终效果 关键词表能够反映出每个关键词与哪些文档段相关联
        return keyword_table

    def _delete_ids_from_keyword_table(self, keyword_table: dict, ids: list[str]) -> dict:
        # get set of ids that correspond to node
        node_idxs_to_delete = set(ids)

        # delete node_idxs from keyword to node idxs mapping
        keywords_to_delete = set()
        for keyword, node_idxs in keyword_table.items():
            if node_idxs_to_delete.intersection(node_idxs):
                keyword_table[keyword] = node_idxs.difference(
                    node_idxs_to_delete
                )
                if not keyword_table[keyword]:
                    keywords_to_delete.add(keyword)

        for keyword in keywords_to_delete:
            del keyword_table[keyword]

        return keyword_table

    def _retrieve_ids_by_query(self, keyword_table: dict, query: str, k: int = 4):
        keyword_table_handler = JiebaKeywordTableHandler()
        # 从查询语句中提取关键词
        keywords = keyword_table_handler.extract_keywords(query)

        # go through text chunks in order of most matching keywords
        # 与关键词表进行匹配,从而 确定 匹配的数据块
        chunk_indices_count: dict[str, int] = defaultdict(int)
        keywords = [keyword for keyword in keywords if keyword in set(keyword_table.keys())]
        for keyword in keywords:
            for node_id in keyword_table[keyword]:
                chunk_indices_count[node_id] += 1

        sorted_chunk_indices = sorted(
            chunk_indices_count.keys(),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )

        return sorted_chunk_indices[: k]

    #更新特定文档段（DocumentSegment）的关键词列表
    def _update_segment_keywords(self, dataset_id: str, node_id: str, keywords: list[str]):
        document_segment = db.session.query(DocumentSegment).filter(
            DocumentSegment.dataset_id == dataset_id,
            DocumentSegment.index_node_id == node_id
        ).first()
        if document_segment:
            document_segment.keywords = keywords
            db.session.add(document_segment)
            db.session.commit()

    def create_segment_keywords(self, node_id: str, keywords: list[str]):
        keyword_table = self._get_dataset_keyword_table()
        self._update_segment_keywords(self.dataset.id, node_id, keywords)
        keyword_table = self._add_text_to_keyword_table(keyword_table, node_id, keywords)
        self._save_dataset_keyword_table(keyword_table)

    def multi_create_segment_keywords(self, pre_segment_data_list: list):
        keyword_table_handler = JiebaKeywordTableHandler()
        keyword_table = self._get_dataset_keyword_table()
        for pre_segment_data in pre_segment_data_list:
            segment = pre_segment_data['segment']
            if pre_segment_data['keywords']:
                segment.keywords = pre_segment_data['keywords']
                keyword_table = self._add_text_to_keyword_table(keyword_table, segment.index_node_id,
                                                                pre_segment_data['keywords'])
            else:
                keywords = keyword_table_handler.extract_keywords(segment.content,
                                                                  self._config.max_keywords_per_chunk)
                segment.keywords = list(keywords)
                keyword_table = self._add_text_to_keyword_table(keyword_table, segment.index_node_id, list(keywords))
        self._save_dataset_keyword_table(keyword_table)

    def update_segment_keywords_index(self, node_id: str, keywords: list[str]):
        keyword_table = self._get_dataset_keyword_table()
        keyword_table = self._add_text_to_keyword_table(keyword_table, node_id, keywords)
        self._save_dataset_keyword_table(keyword_table)


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)
