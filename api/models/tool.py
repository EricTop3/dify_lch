import json
from enum import Enum

from extensions.ext_database import db
from models import StringUUID


class ToolProviderName(Enum):
    SERPAPI = 'serpapi' #表示工具提供商名称的枚举值

    @staticmethod
    def value_of(value):
      # 用于根据提供的值获取对应的枚举成员
        for member in ToolProviderName:
            if member.value == value:
                return member
        raise ValueError(f"No matching enum found for value '{value}'")


class ToolProvider(db.Model):
  # tool_providers 用于存储工具提供商的信息 工具提供商表
    __tablename__ = 'tool_providers'
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='tool_provider_pkey'),
        db.UniqueConstraint('tenant_id', 'tool_name', name='unique_tool_provider_tool_name')
    )

    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))  #自增主键，唯一标识工具提供商记录
    tenant_id = db.Column(StringUUID, nullable=False)                         # 租户的唯一标识符
    tool_name = db.Column(db.String(40), nullable=False)                      #工具名称
    # 工具的加密凭证
    encrypted_credentials = db.Column(db.Text, nullable=True)
    is_enabled = db.Column(db.Boolean, nullable=False, server_default=db.text('false'))     #工具是否启用
    created_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))
    updated_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))

    @property
    def credentials_is_set(self):
        """
         Returns True if the encrypted_config is not None, indicating that the token is set. 返回 True，表示凭证已设置
         """
        return self.encrypted_credentials is not None

    @property
    def credentials(self):
        """
        Returns the decrypted config.
        """
        return json.loads(self.encrypted_credentials) if self.encrypted_credentials is not None else None
