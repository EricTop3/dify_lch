from typing import Optional

from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.errors.invoke import InvokeAuthorizationError
from core.rag.data_post_processor.reorder import ReorderRunner
from core.rag.models.document import Document
from core.rag.rerank.constants.rerank_mode import RerankMode
from core.rag.rerank.entity.weight import KeywordSetting, VectorSetting, Weights
from core.rag.rerank.rerank_model import RerankModelRunner
from core.rag.rerank.weight_rerank import WeightRerankRunner


class DataPostProcessor:
    """Interface for data post-processing document.
    各种检索方式检索到匹配的文档片段 可以视作一个粗筛的过程，
    之后需要在做一个 rerank 从所有检索片段中进一步精筛来获取最匹配的文档片段
    也就是由 DataPostProcessor 来实现
    """

    def __init__(self, tenant_id: str, reranking_mode: str,
                 reranking_model: Optional[dict] = None, weights: Optional[dict] = None,
                 reorder_enabled: bool = False):
        self.rerank_runner = self._get_rerank_runner(reranking_mode, tenant_id, reranking_model, weights)
        self.reorder_runner = self._get_reorder_runner(reorder_enabled)

    def invoke(self, query: str, documents: list[Document], score_threshold: Optional[float] = None,
               top_n: Optional[int] = None, user: Optional[str] = None) -> list[Document]:
        if self.rerank_runner:
            # 调用rerank模型做精筛
            documents = self.rerank_runner.run(query, documents, score_threshold, top_n, user)

        if self.reorder_runner:
            # 将精筛的文档片段做一下乱序，乱序的策略是奇数位+偶数位的倒置 ：[0, 1, 2, 3, 4, 5] -> [0, 2, 4, 5, 3, 1]
            documents = self.reorder_runner.run(documents)

        return documents

    def _get_rerank_runner(self, reranking_mode: str, tenant_id: str, reranking_model: Optional[dict] = None,
                           weights: Optional[dict] = None) -> Optional[RerankModelRunner | WeightRerankRunner]:
        if reranking_mode == RerankMode.WEIGHTED_SCORE.value and weights:
            return WeightRerankRunner(
                tenant_id,
                Weights(
                    vector_setting=VectorSetting(
                        vector_weight=weights['vector_setting']['vector_weight'],
                        embedding_provider_name=weights['vector_setting']['embedding_provider_name'],
                        embedding_model_name=weights['vector_setting']['embedding_model_name'],
                    ),
                    keyword_setting=KeywordSetting(
                        keyword_weight=weights['keyword_setting']['keyword_weight'],
                    )
                )
            )
        elif reranking_mode == RerankMode.RERANKING_MODEL.value:
            if reranking_model:
                try:
                    model_manager = ModelManager()
                    rerank_model_instance = model_manager.get_model_instance(
                        tenant_id=tenant_id,
                        provider=reranking_model['reranking_provider_name'],
                        model_type=ModelType.RERANK,
                        model=reranking_model['reranking_model_name']
                    )
                except InvokeAuthorizationError:
                    return None
                return RerankModelRunner(rerank_model_instance)
            return None
        return None

    def _get_reorder_runner(self, reorder_enabled) -> Optional[ReorderRunner]:
        if reorder_enabled:
            return ReorderRunner()
        return None


