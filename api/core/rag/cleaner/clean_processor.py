import re


class CleanProcessor:

    @classmethod
    def clean(cls, text: str, process_rule: dict) -> str:
        # default clean 默认清理
        # remove invalid symbol
        text = re.sub(r'<\|', '<', text)
        text = re.sub(r'\|>', '>', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\xEF\xBF\xBE]', '', text)
        # Unicode  U+FFFE
        text = re.sub('\uFFFE', '', text)
        # 规则应用 process_rule 字典 包含了一系列的清理规则，这些规则在 rules 键下的 pre_processing_rules 列表中定义
        rules = process_rule['rules'] if process_rule else None
        if 'pre_processing_rules' in rules:
            pre_processing_rules = rules["pre_processing_rules"]
            for pre_processing_rule in pre_processing_rules:
                if pre_processing_rule["id"] == "remove_extra_spaces" and pre_processing_rule["enabled"] is True:
                    # Remove extra spaces 删除额外的空格
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

    def filter_string(self, text):

        return text
