import re
from typing import Optional

import jieba
from jieba.analyse import default_tfidf

from core.rag.datasource.keyword.jieba.stopwords import STOPWORDS


class JiebaKeywordTableHandler:

    def __init__(self):
        default_tfidf.stop_words = STOPWORDS

    def extract_keywords(self, text: str, max_keywords_per_chunk: Optional[int] = 10) -> set[str]:
        """Extract keywords with JIEBA tfidf."""    # 使用 JIEBA tfidf 提取关键词
        keywords = jieba.analyse.extract_tags(
            sentence=text,
            topK=max_keywords_per_chunk,
        )

        return set(self._expand_tokens_with_subtokens(keywords))

    # 目的是从给定的一组关键词中进一步提取子关键词，并过滤掉停用词
    # 这个方法能够帮助提高关键词的覆盖范围和细粒度，从而更准确地反映文本的内容
    def _expand_tokens_with_subtokens(self, tokens: set[str]) -> set[str]:
        """Get subtokens from a list of tokens., filtering for stopwords."""  # 从一组标记中获取子标记，过滤停用词
        results = set()
        for token in tokens:
            results.add(token)                           # 将当前关键词 token 直接添加到结果集 results 中，因为原始关键词本身也是有价值的
            sub_tokens = re.findall(r"\w+", token)       # r"\w+" 可以匹配所有的字母、数字和下划线字符，因此，使用这个模式的 re.findall() 函数会提取字符串中的所有单词
            if len(sub_tokens) > 1:
                results.update({w for w in sub_tokens if w not in list(STOPWORDS)})

        return results