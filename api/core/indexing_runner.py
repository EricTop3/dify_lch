import concurrent.futures
import datetime
import json
import logging
import re
import threading
import time
import uuid
from typing import Optional, cast

from flask import Flask, current_app
from flask_login import current_user
from sqlalchemy.orm.exc import ObjectDeletedError

from configs import dify_config
from core.errors.error import ProviderTokenNotInitError
from core.llm_generator.llm_generator import LLMGenerator
from core.model_manager import ModelInstance, ModelManager
from core.model_runtime.entities.model_entities import ModelType, PriceType
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.model_runtime.model_providers.__base.text_embedding_model import TextEmbeddingModel
from core.rag.datasource.keyword.keyword_factory import Keyword
from core.rag.docstore.dataset_docstore import DatasetDocumentStore
from core.rag.extractor.entity.extract_setting import ExtractSetting
from core.rag.index_processor.index_processor_base import BaseIndexProcessor
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from core.rag.models.document import Document
from core.rag.splitter.fixed_text_splitter import (
    EnhanceRecursiveCharacterTextSplitter,
    FixedRecursiveCharacterTextSplitter,
)
from core.rag.splitter.text_splitter import TextSplitter
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from libs import helper
from models.dataset import Dataset, DatasetProcessRule, DocumentSegment
from models.dataset import Document as DatasetDocument
from models.model import UploadFile
from services.feature_service import FeatureService


# 类中的函数大多数是私有函数，它们通常作为run方法的一部分被调用，或者在估计索引过程时使用。
# 每个函数都专注于索引过程的一个特定方面，从文档的提取 和 清理到分割、转换 和 最终的索引加载

class IndexingRunner:

    def __init__(self):
    #初始化索引运行器， 初始化存储 和 模型管理器
        self.storage = storage
        self.model_manager = ModelManager()
    #运行索引过程 遍历数据集文档，执行 提取、转换、保存段 和 加载步骤
    def run(self, dataset_documents: list[DatasetDocument]):
        """Run the indexing process."""
        for dataset_document in dataset_documents:
            try:
                # get dataset
                dataset = Dataset.query.filter_by(
                    id=dataset_document.dataset_id
                ).first()

                if not dataset:
                    raise ValueError("no dataset found")

                # get the process rule
                processing_rule = db.session.query(DatasetProcessRule). \
                    filter(DatasetProcessRule.id == dataset_document.dataset_process_rule_id). \
                    first()
                index_type = dataset_document.doc_form
                index_processor = IndexProcessorFactory(index_type).init_index_processor()
                # extract 提取文档信息 这里就需要适配各种文档格式，提取各种富文本中的文字信息
                text_docs = self._extract(index_processor, dataset_document, processing_rule.to_dict())

                # transform 将文档内容切片，单个文档 按照 处理规则切分多个 chunks
                documents = self._transform(index_processor, dataset, text_docs, dataset_document.doc_language,
                                            processing_rule.to_dict())
                # save segment 存储片段 将最终切片后的 chunks 构造 document_segment 入库
                self._load_segments(dataset, dataset_document, documents)

                # load
                self._load(
                    index_processor=index_processor,
                    dataset=dataset,
                    dataset_document=dataset_document,
                    documents=documents
                )
            except DocumentIsPausedException:
                raise DocumentIsPausedException('Document paused, document id: {}'.format(dataset_document.id))
            except ProviderTokenNotInitError as e:
                dataset_document.indexing_status = 'error'
                dataset_document.error = str(e.description)
                dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                db.session.commit()
            except ObjectDeletedError:
                logging.warning('Document deleted, document id: {}'.format(dataset_document.id))
            except Exception as e:
                logging.exception("consume document failed")
                dataset_document.indexing_status = 'error'
                dataset_document.error = str(e)
                dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                db.session.commit()
    #当索引状态为拆分时运行索引过程 删除现有文档段，重新提取、转换、保存段和加载
    def run_in_splitting_status(self, dataset_document: DatasetDocument):
        """Run the indexing process when the index_status is splitting."""
        try:
            # get dataset
            dataset = Dataset.query.filter_by(
                id=dataset_document.dataset_id
            ).first()

            if not dataset:
                raise ValueError("no dataset found")

            # get exist document_segment list and delete
            document_segments = DocumentSegment.query.filter_by(
                dataset_id=dataset.id,
                document_id=dataset_document.id
            ).all()

            for document_segment in document_segments:
                db.session.delete(document_segment)
            db.session.commit()
            # get the process rule
            processing_rule = db.session.query(DatasetProcessRule). \
                filter(DatasetProcessRule.id == dataset_document.dataset_process_rule_id). \
                first()

            index_type = dataset_document.doc_form
            index_processor = IndexProcessorFactory(index_type).init_index_processor()
            # extract 提取数据
            text_docs = self._extract(index_processor, dataset_document, processing_rule.to_dict())

            # transform 数据转换 完成数据清理（除了无效符号、多余的空格、URL和电子邮件地址等）、分割等数据加工（给数据增加一些唯一识别码）
            documents = self._transform(index_processor, dataset, text_docs, dataset_document.doc_language,
                                        processing_rule.to_dict())
            # save segment 数据入库 经过前面的处理，数据已经比较合理，到了可以灌库的阶段
            # 不仅支持向量数据库灌库，还支持关键字灌库（到了jieba）
            self._load_segments(dataset, dataset_document, documents)

            # load
            self._load(
                index_processor=index_processor,
                dataset=dataset,
                dataset_document=dataset_document,
                documents=documents
            )
        except DocumentIsPausedException:
            raise DocumentIsPausedException('Document paused, document id: {}'.format(dataset_document.id))
        except ProviderTokenNotInitError as e:
            dataset_document.indexing_status = 'error'
            dataset_document.error = str(e.description)
            dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            db.session.commit()
        except Exception as e:
            logging.exception("consume document failed")
            dataset_document.indexing_status = 'error'
            dataset_document.error = str(e)
            dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            db.session.commit()
    # 当索引状态为索引时运行索引过程 转换未完成的文档段，然后加载它们
    def run_in_indexing_status(self, dataset_document: DatasetDocument):
        """Run the indexing process when the index_status is indexing."""
        try:
            # get dataset
            dataset = Dataset.query.filter_by(
                id=dataset_document.dataset_id
            ).first()

            if not dataset:
                raise ValueError("no dataset found")

            # get exist document_segment list and delete
            document_segments = DocumentSegment.query.filter_by(
                dataset_id=dataset.id,
                document_id=dataset_document.id
            ).all()

            documents = []
            if document_segments:
                for document_segment in document_segments:
                    # transform segment to node
                    if document_segment.status != "completed":
                        document = Document(
                            page_content=document_segment.content,
                            metadata={
                                "doc_id": document_segment.index_node_id,
                                "doc_hash": document_segment.index_node_hash,
                                "document_id": document_segment.document_id,
                                "dataset_id": document_segment.dataset_id,
                            }
                        )

                        documents.append(document)

            # build index
            # get the process rule
            processing_rule = db.session.query(DatasetProcessRule). \
                filter(DatasetProcessRule.id == dataset_document.dataset_process_rule_id). \
                first()

            index_type = dataset_document.doc_form
            index_processor = IndexProcessorFactory(index_type).init_index_processor()
            self._load(
                index_processor=index_processor,
                dataset=dataset,
                dataset_document=dataset_document,
                documents=documents
            )
        except DocumentIsPausedException:
            raise DocumentIsPausedException('Document paused, document id: {}'.format(dataset_document.id))
        except ProviderTokenNotInitError as e:
            dataset_document.indexing_status = 'error'
            dataset_document.error = str(e.description)
            dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            db.session.commit()
        except Exception as e:
            logging.exception("consume document failed")
            dataset_document.indexing_status = 'error'
            dataset_document.error = str(e)
            dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            db.session.commit()

    # 估计文档索引所需的资源 检查文档限制，提取文本文档，分割文档，计算令牌数 和 价格
    def indexing_estimate(self, tenant_id: str, extract_settings: list[ExtractSetting], tmp_processing_rule: dict,
                          doc_form: str = None, doc_language: str = 'English', dataset_id: str = None,
                          indexing_technique: str = 'economy') -> dict:
        """
        Estimate the indexing for the document.
        """
        # check document limit
        features = FeatureService.get_features(tenant_id)
        if features.billing.enabled:
            count = len(extract_settings)
            batch_upload_limit = dify_config.BATCH_UPLOAD_LIMIT
            if count > batch_upload_limit:
                raise ValueError(f"You have reached the batch upload limit of {batch_upload_limit}.")

        embedding_model_instance = None
        if dataset_id:
            dataset = Dataset.query.filter_by(
                id=dataset_id
            ).first()
            if not dataset:
                raise ValueError('Dataset not found.')
            if dataset.indexing_technique == 'high_quality' or indexing_technique == 'high_quality':
                if dataset.embedding_model_provider:
                    # 获取嵌入模型实例
                    embedding_model_instance = self.model_manager.get_model_instance(
                        tenant_id=tenant_id,
                        provider=dataset.embedding_model_provider,
                        model_type=ModelType.TEXT_EMBEDDING,
                        model=dataset.embedding_model
                    )
                else:
                    embedding_model_instance = self.model_manager.get_default_model_instance(
                        tenant_id=tenant_id,
                        model_type=ModelType.TEXT_EMBEDDING,
                    )
        else:
            if indexing_technique == 'high_quality':
                embedding_model_instance = self.model_manager.get_default_model_instance(
                    tenant_id=tenant_id,
                    model_type=ModelType.TEXT_EMBEDDING,
                )
        tokens = 0
        preview_texts = []
        total_segments = 0
        total_price = 0
        currency = 'USD'
        index_type = doc_form
        index_processor = IndexProcessorFactory(index_type).init_index_processor()
        all_text_docs = []
        for extract_setting in extract_settings:
            # extract
            text_docs = index_processor.extract(extract_setting, process_rule_mode=tmp_processing_rule["mode"])
            all_text_docs.extend(text_docs)
            processing_rule = DatasetProcessRule(
                mode=tmp_processing_rule["mode"],
                rules=json.dumps(tmp_processing_rule["rules"])
            )

            # get splitter
            splitter = self._get_splitter(processing_rule, embedding_model_instance)

            # split to documents
            documents = self._split_to_documents_for_estimate(
                text_docs=text_docs,
                splitter=splitter,
                processing_rule=processing_rule
            )

            total_segments += len(documents)
            for document in documents:
                if len(preview_texts) < 5:
                    preview_texts.append(document.page_content)
                if indexing_technique == 'high_quality' or embedding_model_instance:
                    tokens += embedding_model_instance.get_text_embedding_num_tokens(
                        texts=[self.filter_string(document.page_content)]
                    )

        if doc_form and doc_form == 'qa_model':
            model_instance = self.model_manager.get_default_model_instance(
                tenant_id=tenant_id,
                model_type=ModelType.LLM
            )

            model_type_instance = model_instance.model_type_instance
            model_type_instance = cast(LargeLanguageModel, model_type_instance)

            if len(preview_texts) > 0:
                # qa model document
                response = LLMGenerator.generate_qa_document(current_user.current_tenant_id, preview_texts[0],
                                                             doc_language)
                document_qa_list = self.format_split_text(response)
                price_info = model_type_instance.get_price(
                    model=model_instance.model,
                    credentials=model_instance.credentials,
                    price_type=PriceType.INPUT,
                    tokens=total_segments * 2000,
                )
                return {
                    "total_segments": total_segments * 20,
                    "tokens": total_segments * 2000,
                    "total_price": '{:f}'.format(price_info.total_amount),
                    "currency": price_info.currency,
                    "qa_preview": document_qa_list,
                    "preview": preview_texts
                }
        if embedding_model_instance:
            # 转换为 TextEmbeddingModel 对象
            embedding_model_type_instance = cast(TextEmbeddingModel, embedding_model_instance.model_type_instance)
            embedding_price_info = embedding_model_type_instance.get_price(
                model=embedding_model_instance.model,
                credentials=embedding_model_instance.credentials,
                price_type=PriceType.INPUT,
                tokens=tokens
            )
            total_price = '{:f}'.format(embedding_price_info.total_amount)
            currency = embedding_price_info.currency
        return {
            "total_segments": total_segments,
            "tokens": tokens,
            "total_price": total_price,
            "currency": currency,
            "preview": preview_texts
        }

    # 提取文档内容 根据数据源类型加载文件 或 Notion 信息，并提取文本
    def _extract(self, index_processor: BaseIndexProcessor, dataset_document: DatasetDocument, process_rule: dict) \
            -> list[Document]:
        # load file
        if dataset_document.data_source_type not in ["upload_file", "notion_import", "website_crawl"]: # 数据源类型
            return []

        data_source_info = dataset_document.data_source_info_dict       # 数据源信息
        text_docs = []
        if dataset_document.data_source_type == 'upload_file':
            if not data_source_info or 'upload_file_id' not in data_source_info:
                raise ValueError("no upload file found")

            file_detail = db.session.query(UploadFile). \
                filter(UploadFile.id == data_source_info['upload_file_id']). \
                one_or_none()

            if file_detail:
                extract_setting = ExtractSetting(
                    datasource_type="upload_file",
                    upload_file=file_detail,
                    document_model=dataset_document.doc_form
                )
                text_docs = index_processor.extract(extract_setting, process_rule_mode=process_rule['mode'])
        elif dataset_document.data_source_type == 'notion_import':
            if (not data_source_info or 'notion_workspace_id' not in data_source_info
                    or 'notion_page_id' not in data_source_info):
                raise ValueError("no notion import info found")
            extract_setting = ExtractSetting(
                datasource_type="notion_import",
                notion_info={
                    "notion_workspace_id": data_source_info['notion_workspace_id'],
                    "notion_obj_id": data_source_info['notion_page_id'],
                    "notion_page_type": data_source_info['type'],
                    "document": dataset_document,
                    "tenant_id": dataset_document.tenant_id
                },
                document_model=dataset_document.doc_form
            )
            text_docs = index_processor.extract(extract_setting, process_rule_mode=process_rule['mode'])
        elif dataset_document.data_source_type == 'website_crawl':
            if (not data_source_info or 'provider' not in data_source_info
                    or 'url' not in data_source_info or 'job_id' not in data_source_info):
                raise ValueError("no website import info found")
            extract_setting = ExtractSetting(
                datasource_type="website_crawl",
                website_info={
                    "provider": data_source_info['provider'],
                    "job_id": data_source_info['job_id'],
                    "tenant_id": dataset_document.tenant_id,
                    "url": data_source_info['url'],
                    "mode": data_source_info['mode'],
                    "only_main_content": data_source_info['only_main_content']
                },
                document_model=dataset_document.doc_form
            )
            text_docs = index_processor.extract(extract_setting, process_rule_mode=process_rule['mode'])
        # update document status to splitting
        self._update_document_index_status(
            document_id=dataset_document.id,
            after_indexing_status="splitting",  # 更新文档状态为拆分
            extra_update_params={
                DatasetDocument.word_count: sum(len(text_doc.page_content) for text_doc in text_docs),
                DatasetDocument.parsing_completed_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            }
        )

        # replace doc id to document model id
        text_docs = cast(list[Document], text_docs)
        for text_doc in text_docs:
            text_doc.metadata['document_id'] = dataset_document.id
            text_doc.metadata['dataset_id'] = dataset_document.dataset_id

        return text_docs
    # 过滤字符串，移除不需要的字符 使用正则表达式 替换 或 删除 文本中的特定字符
    def filter_string(self, text):
        text = re.sub(r'<\|', '<', text)
        text = re.sub(r'\|>', '>', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\xEF\xBF\xBE]', '', text)
        # Unicode  U+FFFE
        text = re.sub('\uFFFE', '', text)
        return text

    # 获取分割器 根据处理规则获取文本分割器
    def _get_splitter(self, processing_rule: DatasetProcessRule,
                      embedding_model_instance: Optional[ModelInstance]) -> TextSplitter:
        """
        Get the NodeParser object according to the processing rule.
        """
        if processing_rule.mode == "custom":
            # The user-defined segmentation rule 自定义
            rules = json.loads(processing_rule.rules)
            segmentation = rules["segmentation"]
            max_segmentation_tokens_length = dify_config.INDEXING_MAX_SEGMENTATION_TOKENS_LENGTH
            if segmentation["max_tokens"] < 50 or segmentation["max_tokens"] > max_segmentation_tokens_length:
                raise ValueError(f"Custom segment length should be between 50 and {max_segmentation_tokens_length}.")

            separator = segmentation["separator"]
            if separator:
                separator = separator.replace('\\n', '\n')

            if segmentation.get('chunk_overlap'):
                chunk_overlap = segmentation['chunk_overlap']
            else:
                chunk_overlap = 0

            character_splitter = FixedRecursiveCharacterTextSplitter.from_encoder(
                chunk_size=segmentation["max_tokens"],
                chunk_overlap=chunk_overlap,
                fixed_separator=separator,
                separators=["\n\n", "。", ". ", " ", ""],
                embedding_model_instance=embedding_model_instance
            )
        else:
            # Automatic segmentation 自动分段
            character_splitter = EnhanceRecursiveCharacterTextSplitter.from_encoder(
                chunk_size=DatasetProcessRule.AUTOMATIC_RULES['segmentation']['max_tokens'],
                chunk_overlap=DatasetProcessRule.AUTOMATIC_RULES['segmentation']['chunk_overlap'],
                separators=["\n\n", "。", ". ", " ", ""],
                embedding_model_instance=embedding_model_instance
            )

        return character_splitter
    # 步骤拆分 将文本文档拆分成 文档段 并 保存
    def _step_split(self, text_docs: list[Document], splitter: TextSplitter,
                    dataset: Dataset, dataset_document: DatasetDocument, processing_rule: DatasetProcessRule) \
            -> list[Document]:
        """
        Split the text documents into documents and save them to the document segment.
        """
        documents = self._split_to_documents(
            text_docs=text_docs,
            splitter=splitter,
            processing_rule=processing_rule,
            tenant_id=dataset.tenant_id,
            document_form=dataset_document.doc_form,
            document_language=dataset_document.doc_language
        )

        # save node to document segment
        doc_store = DatasetDocumentStore(
            dataset=dataset,
            user_id=dataset_document.created_by,
            document_id=dataset_document.id
        )

        # add document segments
        doc_store.add_documents(documents)

        # update document status to indexing
        cur_time = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        self._update_document_index_status(
            document_id=dataset_document.id,
            after_indexing_status="indexing",
            extra_update_params={
                DatasetDocument.cleaning_completed_at: cur_time,
                DatasetDocument.splitting_completed_at: cur_time,
            }
        )

        # update segment status to indexing
        self._update_segments_by_document(
            dataset_document_id=dataset_document.id,
            update_params={
                DocumentSegment.status: "indexing",
                DocumentSegment.indexing_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            }
        )

        return documents
    # 拆分文档为多个节点 使用分割器将文本拆分成多个文档节点
    def _split_to_documents(self, text_docs: list[Document], splitter: TextSplitter,
                            processing_rule: DatasetProcessRule, tenant_id: str,
                            document_form: str, document_language: str) -> list[Document]:
        """
        Split the text documents into nodes.
        """
        all_documents = []
        all_qa_documents = []
        for text_doc in text_docs:
            # document clean
            document_text = self._document_clean(text_doc.page_content, processing_rule)
            text_doc.page_content = document_text

            # parse document to nodes
            documents = splitter.split_documents([text_doc])
            split_documents = []
            for document_node in documents:

                if document_node.page_content.strip():
                    doc_id = str(uuid.uuid4())
                    hash = helper.generate_text_hash(document_node.page_content)
                    document_node.metadata['doc_id'] = doc_id
                    document_node.metadata['doc_hash'] = hash
                    # delete Spliter character
                    page_content = document_node.page_content
                    if page_content.startswith(".") or page_content.startswith("。"):
                        page_content = page_content[1:]
                    else:
                        page_content = page_content
                    document_node.page_content = page_content

                    if document_node.page_content:
                        split_documents.append(document_node)
            all_documents.extend(split_documents)
        # processing qa document
        if document_form == 'qa_model':
            for i in range(0, len(all_documents), 10):
                threads = []
                sub_documents = all_documents[i:i + 10]
                for doc in sub_documents:
                    document_format_thread = threading.Thread(target=self.format_qa_document, kwargs={
                        'flask_app': current_app._get_current_object(),
                        'tenant_id': tenant_id, 'document_node': doc, 'all_qa_documents': all_qa_documents,
                        'document_language': document_language})
                    threads.append(document_format_thread)
                    document_format_thread.start()
                for thread in threads:
                    thread.join()
            return all_qa_documents
        return all_documents
    # 格式化问答模型文档 对于问答模型，生成问题和答案对
    def format_qa_document(self, flask_app: Flask, tenant_id: str, document_node, all_qa_documents, document_language):
        format_documents = []
        if document_node.page_content is None or not document_node.page_content.strip():
            return
        with flask_app.app_context():
            try:
                # qa model document
                response = LLMGenerator.generate_qa_document(tenant_id, document_node.page_content, document_language)
                document_qa_list = self.format_split_text(response)
                qa_documents = []
                for result in document_qa_list:
                    qa_document = Document(page_content=result['question'], metadata=document_node.metadata.model_copy())
                    doc_id = str(uuid.uuid4())
                    hash = helper.generate_text_hash(result['question'])
                    qa_document.metadata['answer'] = result['answer']
                    qa_document.metadata['doc_id'] = doc_id
                    qa_document.metadata['doc_hash'] = hash
                    qa_documents.append(qa_document)
                format_documents.extend(qa_documents)
            except Exception as e:
                logging.exception(e)

            all_qa_documents.extend(format_documents)

    # 为估计过程拆分文档节点 拆分文档以估计索引过程所需的资源
    def _split_to_documents_for_estimate(self, text_docs: list[Document], splitter: TextSplitter,
                                         processing_rule: DatasetProcessRule) -> list[Document]:
        """
        Split the text documents into nodes.
        """
        all_documents = []
        for text_doc in text_docs:
            # document clean
            document_text = self._document_clean(text_doc.page_content, processing_rule)
            text_doc.page_content = document_text

            # parse document to nodes
            documents = splitter.split_documents([text_doc])

            split_documents = []
            for document in documents:
                if document.page_content is None or not document.page_content.strip():
                    continue
                doc_id = str(uuid.uuid4())
                hash = helper.generate_text_hash(document.page_content)

                document.metadata['doc_id'] = doc_id
                document.metadata['doc_hash'] = hash

                split_documents.append(document)

            all_documents.extend(split_documents)

        return all_documents
    # 根据处理规则清理文档内容 移除额外的空格、URL和电子邮件等
    def _document_clean(self, text: str, processing_rule: DatasetProcessRule) -> str:
        """
        Clean the document text according to the processing rules.
        """
        if processing_rule.mode == "automatic":
            rules = DatasetProcessRule.AUTOMATIC_RULES
        else:
            rules = json.loads(processing_rule.rules) if processing_rule.rules else {}

        if 'pre_processing_rules' in rules:
            pre_processing_rules = rules["pre_processing_rules"]
            for pre_processing_rule in pre_processing_rules:
                if pre_processing_rule["id"] == "remove_extra_spaces" and pre_processing_rule["enabled"] is True:
                    # Remove extra spaces
                    pattern = r'\n{3,}'
                    text = re.sub(pattern, '\n\n', text)
                    pattern = r'[\t\f\r\x20\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]{2,}'
                    text = re.sub(pattern, ' ', text)
                elif pre_processing_rule["id"] == "remove_urls_emails" and pre_processing_rule["enabled"] is True:
                    # Remove email
                    pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
                    text = re.sub(pattern, '', text)

                    # Remove URL
                    pattern = r'https?://[^\s]+'
                    text = re.sub(pattern, '', text)

        return text
    # 格式化拆分文本 从文本中提取问题和答案对
    def format_split_text(self, text):
        regex = r"Q\d+:\s*(.*?)\s*A\d+:\s*([\s\S]*?)(?=Q\d+:|$)"
        matches = re.findall(regex, text, re.UNICODE)

        return [
            {
                "question": q,
                "answer": re.sub(r"\n\s*", "\n", a.strip())
            }
            for q, a in matches if q and a
        ]
    # 将处理后的文档数据加载到索引中，并更新相关文档 和 文档段 的状态为完成，以支持后续的 搜索 和 检索 操作
    def _load(self, index_processor: BaseIndexProcessor, dataset: Dataset,
              dataset_document: DatasetDocument, documents: list[Document]) -> None:
        """
        insert index and update document/segment status to completed
        """

        embedding_model_instance = None
        if dataset.indexing_technique == 'high_quality':
            # 获取嵌入模型实例
            embedding_model_instance = self.model_manager.get_model_instance(
                tenant_id=dataset.tenant_id,
                provider=dataset.embedding_model_provider,
                model_type=ModelType.TEXT_EMBEDDING,
                model=dataset.embedding_model
            )

        # chunk nodes by chunk size
        indexing_start_at = time.perf_counter()
        tokens = 0
        chunk_size = 10

        # create keyword index 创建关键字索引
        # 在一个独立的线程中，对文档数据进行 关键字索引的创建，以便于后续的搜索和检索
        create_keyword_thread = threading.Thread(target=self._process_keyword_index,
                                                 args=(current_app._get_current_object(),
                                                       dataset.id, dataset_document.id, documents))
        create_keyword_thread.start()    # 启动线程
        if dataset.indexing_technique == 'high_quality':
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # 创建线程池执行器，以便在多个线程中并发执行任务 max_workers=10 表示线程池中最多可以有10个并发线程
                futures = []                                            #用于存储每个提交的任务的 future 对象
                for i in range(0, len(documents), chunk_size):          # 遍历文档
                    chunk_documents = documents[i:i + chunk_size]       # 块文档
                    # executor.submit()提交一个任务给线程池执行器，每个任务调用 self._process_chunk 方法，并传入一系列参数
                    futures.append(executor.submit(self._process_chunk, current_app._get_current_object(), index_processor,
                                                   chunk_documents, dataset,
                                                   dataset_document, embedding_model_instance))

                for future in futures:            # 遍历所有已提交任务的 future 对象
                    tokens += future.result()     # 调用 future.result() 方法获取任务的结果，并将结果累加到 tokens 变量中。future.result() 会阻塞当前线程，直到任务完成并返回结果 

        create_keyword_thread.join()   # 等待线程结束  调用这个方法会使得调用它的线程（如主线程）等待
        indexing_end_at = time.perf_counter()

        # update document status to completed 更新文档状态
        self._update_document_index_status(
            document_id=dataset_document.id,
            after_indexing_status="completed", #在所有文档数据被成功加载到索引后，更新相关文档的状态为completed，表示索引过程已完成
            extra_update_params={
                DatasetDocument.tokens: tokens,
                DatasetDocument.completed_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                DatasetDocument.indexing_latency: indexing_end_at - indexing_start_at,
            }
        )
    # 处理关键字索引 提取文档中的关键字创建关键字索引 和相关文档段状态的更新
    # 目的是在 Flask 应用的上下文中处理关键字索引
    # 这个方法接收四个参数：flask_app（Flask 应用实例），dataset_id（数据集的 ID），document_id（文档的 ID），以及documents（文档对象列表）
    def _process_keyword_index(self, flask_app, dataset_id, document_id, documents):
        # 使用 Flask 应用的上下文：这是必要的步骤，因为在 Flask 应用之外的线程中执行数据库操作或者访问 Flask 应用的配置时，需要手动创建应用上下文
        with flask_app.app_context():
            dataset = Dataset.query.filter_by(id=dataset_id).first()
            if not dataset:
                raise ValueError("no dataset found")
            # 创建关键字索引 使用查询到的数据集对象初始化 Keyword 类的实例，并调用其 create 方法，传入文档对象列表来创建关键字索引
            keyword = Keyword(dataset)
            keyword.create(documents)
            if dataset.indexing_technique != 'high_quality':
                document_ids = [document.metadata['doc_id'] for document in documents]
                db.session.query(DocumentSegment).filter(
                    DocumentSegment.document_id == document_id,
                    DocumentSegment.index_node_id.in_(document_ids),
                    DocumentSegment.status == "indexing"
                ).update({
                    DocumentSegment.status: "completed",
                    DocumentSegment.enabled: True,
                    DocumentSegment.completed_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                })

                db.session.commit()

    # 处理文档块的 索引加载 对文档块进行 索引处理，并更新数据库状态
    # 目的是在处理文档块时，计算文档的 tokens 数量，加载索引，并更新数据库中的文档段状态
    def _process_chunk(self, flask_app, index_processor, chunk_documents, dataset, dataset_document,
                       embedding_model_instance):
        # flask_app：Flask 应用对象
        # index_processor：索引处理器对象
        # chunk_documents：要处理的文档块
        # dataset：数据集对象
        # dataset_document：数据集中的文档对象
        # embedding_model_instance：嵌入模型实例

        with flask_app.app_context():    #在 Flask 应用上下文中运行代码。这使得代码可以访问 Flask 的应用配置和数据库连接
            # check document is paused  检查文档是否暂停
            self._check_document_paused_status(dataset_document.id)
            # 计算tokens的数量
            tokens = 0
            if embedding_model_instance:
                tokens += sum(
                    embedding_model_instance.get_text_embedding_num_tokens(
                        [document.page_content]
                    )
                    for document in chunk_documents
                )

            # load index 加载索引
            index_processor.load(dataset, chunk_documents, with_keywords=False)   # with_keywords=False 表示不使用关键字

            document_ids = [document.metadata['doc_id'] for document in chunk_documents]  # 从文档的元数据中提取 doc_id，并生成一个文档 ID 列表
            db.session.query(DocumentSegment).filter(
                DocumentSegment.document_id == dataset_document.id,
                DocumentSegment.index_node_id.in_(document_ids),
                DocumentSegment.status == "indexing"
            ).update({
                DocumentSegment.status: "completed",
                DocumentSegment.enabled: True,
                DocumentSegment.completed_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            })     # 更新文档段

            db.session.commit()    # 提交

            return tokens
    # 检查文档是否已暂停索引 检查Redis缓存中是否有文档暂停的标记
    def _check_document_paused_status(self, document_id: str):
        indexing_cache_key = 'document_{}_is_paused'.format(document_id)
        result = redis_client.get(indexing_cache_key)
        if result:
            raise DocumentIsPausedException()
    # 更新文档索引状态 在数据库中更新文档的索引状态
    def _update_document_index_status(self, document_id: str, after_indexing_status: str,
                                      extra_update_params: Optional[dict] = None) -> None:
        """
        Update the document indexing status.
        """
        count = DatasetDocument.query.filter_by(id=document_id, is_paused=True).count()
        if count > 0:
            raise DocumentIsPausedException()
        document = DatasetDocument.query.filter_by(id=document_id).first()
        if not document:
            raise DocumentIsDeletedPausedException()

        update_params = {
            DatasetDocument.indexing_status: after_indexing_status
        }

        if extra_update_params:
            update_params.update(extra_update_params)

        DatasetDocument.query.filter_by(id=document_id).update(update_params)
        db.session.commit()
    # 根据文档ID更新段的状态 更新与文档ID相关的所有段的状态
    def _update_segments_by_document(self, dataset_document_id: str, update_params: dict) -> None:
        """
        Update the document segment by document id.
        """
        DocumentSegment.query.filter_by(document_id=dataset_document_id).update(update_params)
        db.session.commit()

    # 批量添加 段索引处理 将多个段批量添加到索引中
    def batch_add_segments(self, segments: list[DocumentSegment], dataset: Dataset):
        """
        Batch add segments index processing
        """
        documents = []
        for segment in segments:
            document = Document(
                page_content=segment.content,
                metadata={
                    "doc_id": segment.index_node_id,
                    "doc_hash": segment.index_node_hash,
                    "document_id": segment.document_id,
                    "dataset_id": segment.dataset_id,
                }
            )
            documents.append(document)
        # save vector index
        index_type = dataset.doc_form
        index_processor = IndexProcessorFactory(index_type).init_index_processor()
        index_processor.load(dataset, documents)

    # 转换文档 将提取的文本文档转换为可以索引的格式
    def _transform(self, index_processor: BaseIndexProcessor, dataset: Dataset,
                   text_docs: list[Document], doc_language: str, process_rule: dict) -> list[Document]:
        # get embedding model instance
        embedding_model_instance = None
        if dataset.indexing_technique == 'high_quality':
            if dataset.embedding_model_provider:
                # 获取嵌入模型实例
                embedding_model_instance = self.model_manager.get_model_instance(
                    tenant_id=dataset.tenant_id,
                    provider=dataset.embedding_model_provider,
                    model_type=ModelType.TEXT_EMBEDDING,
                    model=dataset.embedding_model
                )
            else:
                embedding_model_instance = self.model_manager.get_default_model_instance(
                    tenant_id=dataset.tenant_id,
                    model_type=ModelType.TEXT_EMBEDDING,
                )

        documents = index_processor.transform(text_docs, embedding_model_instance=embedding_model_instance,
                                              process_rule=process_rule, tenant_id=dataset.tenant_id,
                                              doc_language=doc_language)

        return documents

    # 将处理后的文档段（ documents ）保存到数据库中，并更新相关文档的状态
    def _load_segments(self, dataset, dataset_document, documents):
        # save node to document segment
        # 初始化数据集文档存储：创建一个 DatasetDocumentStore 实例，这个实例 与 特定的数据集、创建者和文档ID相关联
        doc_store = DatasetDocumentStore(
            dataset=dataset,
            user_id=dataset_document.created_by,
            document_id=dataset_document.id
        )

        # add document segments 添加文档段  documents是一个包含多个文档段的列表
        # 每个文档段都是一个 Document 实例
        doc_store.add_documents(documents)

        # update document status to indexing 更新 文档状态 为索引中
        cur_time = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        self._update_document_index_status(
            document_id=dataset_document.id,
            after_indexing_status="indexing",                       # 在所有文档段都保存到数据库之后，更新原始文档的状态为“索引中”（indexing）
            extra_update_params={
                DatasetDocument.cleaning_completed_at: cur_time,    # 更新文档的清理完成时间和拆分完成时间
                DatasetDocument.splitting_completed_at: cur_time,
            }
        )

        # update segment status to indexing 更新 段状态 为索引中
        self._update_segments_by_document(
            dataset_document_id=dataset_document.id,
            update_params={
                DocumentSegment.status: "indexing",
                DocumentSegment.indexing_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            }
        )
        pass


class DocumentIsPausedException(Exception):
    pass


class DocumentIsDeletedPausedException(Exception):
    pass
