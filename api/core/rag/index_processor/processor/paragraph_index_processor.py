"""Paragraph index processor."""
import uuid
from typing import Optional

from core.rag.cleaner.clean_processor import CleanProcessor
from core.rag.datasource.keyword.keyword_factory import Keyword
from core.rag.datasource.retrieval_service import RetrievalService
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.extractor.entity.extract_setting import ExtractSetting
from core.rag.extractor.extract_processor import ExtractProcessor
from core.rag.index_processor.index_processor_base import BaseIndexProcessor
from core.rag.models.document import Document
from libs import helper
from models.dataset import Dataset


class ParagraphIndexProcessor(BaseIndexProcessor):

    def extract(self, extract_setting: ExtractSetting, **kwargs) -> list[Document]:

        text_docs = ExtractProcessor.extract(extract_setting=extract_setting,
                                             is_automatic=kwargs.get('process_rule_mode') == "automatic")

        return text_docs
    # 对文档进行 预处理 和 分割，以便后续的 索引 或 其它处理步骤 可以更有效地处理文档的各个部分
    def transform(self, documents: list[Document], **kwargs) -> list[Document]:
        # Split the text documents into nodes.
        # 获取分割器：根据传入的处理规则 process_rule 和 嵌入模型实例 embedding_model_instance 获取文档分割器（splitter）
        splitter = self._get_splitter(processing_rule=kwargs.get('process_rule'),
                                      embedding_model_instance=kwargs.get('embedding_model_instance'))
        all_documents = []
        for document in documents:
            # document clean 文档清理、移除不需要的字符或格式
            document_text = CleanProcessor.clean(document.page_content, kwargs.get('process_rule'))
            document.page_content = document_text
            # parse document to nodes 文档分割，将清理后的文档内容分割成多个节点，每个节点代表文档的一部分
            document_nodes = splitter.split_documents([document])
            split_documents = []
            for document_node in document_nodes:
                #节点处理：对于每个分割后的节点，如果节点内容非空，则生成一个唯一的文档ID（ doc_id ）和 文本哈希值（ hash ），
                # 并将这些信息添加到节点的 元数据 中。如果节点内容以特定字符（如.或。）开头，则移除这些字符
                if document_node.page_content.strip():
                    doc_id = str(uuid.uuid4())
                    hash = helper.generate_text_hash(document_node.page_content)
                    document_node.metadata['doc_id'] = doc_id
                    document_node.metadata['doc_hash'] = hash
                    # delete Spliter character
                    page_content = document_node.page_content
                    if page_content.startswith(".") or page_content.startswith("。"):
                        page_content = page_content[1:].strip()
                    else:
                        page_content = page_content
                    if len(page_content) > 0:
                        document_node.page_content = page_content
                        # 收集文档节点 将处理后的节点添加到一个列表中（split_documents），以便进一步处理
                        split_documents.append(document_node)
            all_documents.extend(split_documents)
        return all_documents

    def load(self, dataset: Dataset, documents: list[Document], with_keywords: bool = True):
        if dataset.indexing_technique == 'high_quality':
            vector = Vector(dataset)
            vector.create(documents)
        if with_keywords:
            keyword = Keyword(dataset)
            keyword.create(documents)

    def clean(self, dataset: Dataset, node_ids: Optional[list[str]], with_keywords: bool = True):
        if dataset.indexing_technique == 'high_quality':
            vector = Vector(dataset)
            if node_ids:
                vector.delete_by_ids(node_ids)
            else:
                vector.delete()
        if with_keywords:
            keyword = Keyword(dataset)
            if node_ids:
                keyword.delete_by_ids(node_ids)
            else:
                keyword.delete()

    def retrieve(self, retrival_method: str, query: str, dataset: Dataset, top_k: int,
                 score_threshold: float, reranking_model: dict) -> list[Document]:
        # Set search parameters.
        results = RetrievalService.retrieve(retrival_method=retrival_method, dataset_id=dataset.id, query=query,
                                            top_k=top_k, score_threshold=score_threshold,
                                            reranking_model=reranking_model)
        # Organize results.
        docs = []
        for result in results:
            metadata = result.metadata
            metadata['score'] = result.score
            if result.score > score_threshold:
                doc = Document(page_content=result.page_content, metadata=metadata)
                docs.append(doc)
        return docs
