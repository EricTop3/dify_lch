import datetime
import logging
import time

import click
from celery import shared_task

from configs import dify_config
from core.indexing_runner import DocumentIsPausedException, IndexingRunner
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from extensions.ext_database import db
from models.dataset import Dataset, Document, DocumentSegment
from services.feature_service import FeatureService


@shared_task(queue='dataset')
def duplicate_document_indexing_task(dataset_id: str, document_ids: list):
    """
    Async process document
    :param dataset_id:
    :param document_ids:

    Usage: duplicate_document_indexing_task.delay(dataset_id, document_id)
    """
    # 这个任务用于处理 重复文档 的索引。当已存在的文档被标记为 重复并需要重新索引时，
    # 这个任务会被触发。它接收两个参数：数据集的ID 和 需要重新索引的文档的ID列表。
    # 这个任务会对每个需要重新索引的文档进行索引处理。

    documents = []
    start_at = time.perf_counter()
    # 步骤一 获取指定ID的数据集
    dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()

    # check document limit
    # 步骤二 检查文档的数量是否超过了批量上传的限制或者文档上传的配额。
    # 如果超过了限制，它会抛出一个异常，并将索引状态设置为 error
    features = FeatureService.get_features(dataset.tenant_id)
    try:
        if features.billing.enabled:
            vector_space = features.vector_space
            count = len(document_ids)
            batch_upload_limit = int(dify_config.BATCH_UPLOAD_LIMIT)
            if count > batch_upload_limit:
                raise ValueError(f"You have reached the batch upload limit of {batch_upload_limit}.")
            if 0 < vector_space.limit <= vector_space.size:
                raise ValueError("Your total number of documents plus the number of uploads have over the limit of "
                                 "your subscription.")
    except Exception as e:
        for document_id in document_ids:
            document = db.session.query(Document).filter(
                Document.id == document_id,
                Document.dataset_id == dataset_id
            ).first()
            if document:
                document.indexing_status = 'error'
                document.error = str(e)
                document.stopped_at = datetime.datetime.utcnow()
                db.session.add(document)
        db.session.commit()
        return

    for document_id in document_ids:
        logging.info(click.style('Start process document: {}'.format(document_id), fg='green'))

        document = db.session.query(Document).filter(
            Document.id == document_id,
            Document.dataset_id == dataset_id
        ).first()
        # 步骤三 对每个需要重新索引的文档， 清理旧的数据， 包括从向量索引中 删除旧的数据 和 删除数据库中的旧数据。
        # 然后，将索引状态设置为  parsing  ，并将处理开始的时间设置为当前时间
        if document:
            # clean old data
            index_type = document.doc_form
            index_processor = IndexProcessorFactory(index_type).init_index_processor()

            segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id == document_id).all()
            if segments:
                index_node_ids = [segment.index_node_id for segment in segments]

                # delete from vector index
                index_processor.clean(dataset, index_node_ids)

                for segment in segments:
                    db.session.delete(segment)
                db.session.commit()

            document.indexing_status = 'parsing'
            document.processing_started_at = datetime.datetime.utcnow()
            documents.append(document)
            db.session.add(document)
    db.session.commit()

    try:
        # 步骤四 创建一个 IndexingRunner 实例，并调用其 run 方法来运行索引处理。如果在处理过程中发生了任何异常，捕获这个异常并记录日志
        indexing_runner = IndexingRunner()
        indexing_runner.run(documents)
        end_at = time.perf_counter()
        logging.info(click.style('Processed dataset: {} latency: {}'.format(dataset_id, end_at - start_at), fg='green'))
    except DocumentIsPausedException as ex:
        logging.info(click.style(str(ex), fg='yellow'))
    except Exception:
        pass
