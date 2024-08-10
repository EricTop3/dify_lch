import json

from sqlalchemy import ForeignKey

from core.tools.entities.common_entities import I18nObject
from core.tools.entities.tool_bundle import ApiToolBundle
from core.tools.entities.tool_entities import ApiProviderSchemaType, WorkflowToolParameterConfiguration
from extensions.ext_database import db
from models import StringUUID
from models.model import Account, App, Tenant


class BuiltinToolProvider(db.Model):
    """
    This table stores the tool provider information for built-in tools for each tenant.
    """
    # 工具内置提供商表
    __tablename__ = 'tool_builtin_providers'
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='tool_builtin_provider_pkey'),
        # one tenant can only have one tool provider with the same name
        db.UniqueConstraint('tenant_id', 'provider', name='unique_builtin_tool_provider')
    )

    # id of the tool provider
    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))
    # id of the tenant
    tenant_id = db.Column(StringUUID, nullable=True)
    # who created this tool provider
    user_id = db.Column(StringUUID, nullable=False)
    # name of the tool provider
    provider = db.Column(db.String(40), nullable=False)
    # credential of the tool provider
    encrypted_credentials = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))
    updated_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))

    @property
    def credentials(self) -> dict:
        return json.loads(self.encrypted_credentials)

class PublishedAppTool(db.Model):
    """
    The table stores the apps published as a tool for each person.
    """
    # 工具发布应用表
    __tablename__ = 'tool_published_apps'
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='published_app_tool_pkey'),
        db.UniqueConstraint('app_id', 'user_id', name='unique_published_app_tool')
    )

    # id of the tool provider
    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))
    # id of the app    应用ID
    app_id = db.Column(StringUUID, ForeignKey('apps.id'), nullable=False)
    # who published this tool  用户ID
    user_id = db.Column(StringUUID, nullable=False)
    # description of the tool, stored in i18n format, for human 描述
    description = db.Column(db.Text, nullable=False)
    # llm_description of the tool, for LLM         LLM描述
    llm_description = db.Column(db.Text, nullable=False)
    # query description, query will be seem as a parameter of the tool, to describe this parameter to llm, we need this field
    # 查询描述
    query_description = db.Column(db.Text, nullable=False)
    # query name, the name of the query parameter  查询名称
    query_name = db.Column(db.String(40), nullable=False)
    # name of the tool provider  工具名称
    tool_name = db.Column(db.String(40), nullable=False)
    # author  作者
    author = db.Column(db.String(40), nullable=False)
    # 创建时间
    created_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))
    # 更新时间
    updated_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))

    @property
    def description_i18n(self) -> I18nObject:
        return I18nObject(**json.loads(self.description))
    
    @property
    def app(self) -> App:
        return db.session.query(App).filter(App.id == self.app_id).first()

class ApiToolProvider(db.Model):
    """
    The table stores the api providers.
    """
    # 工具 API 提供商表
    __tablename__ = 'tool_api_providers'
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='tool_api_provider_pkey'),
        db.UniqueConstraint('name', 'tenant_id', name='unique_api_tool_provider')
    )
    # ID
    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))
    # name of the api provider  API提供者名称
    name = db.Column(db.String(40), nullable=False)
    # icon 图标
    icon = db.Column(db.String(255), nullable=False)
    # original schema 模式
    schema = db.Column(db.Text, nullable=False)
    # 模式类型字符串
    schema_type_str = db.Column(db.String(40), nullable=False)
    # who created this tool 用户ID
    user_id = db.Column(StringUUID, nullable=False)
    # tenant id  租户ID
    tenant_id = db.Column(StringUUID, nullable=False)
    # description of the provider 描述
    description = db.Column(db.Text, nullable=False)
    # json format tools 工具字符串
    tools_str = db.Column(db.Text, nullable=False)
    # json format credentials  凭证字符串
    credentials_str = db.Column(db.Text, nullable=False)
    # privacy policy 隐私政策
    privacy_policy = db.Column(db.String(255), nullable=True)
    # custom_disclaimer 自定义免责声明
    custom_disclaimer = db.Column(db.String(255), nullable=True)
    # 创建时间
    created_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))
    # 更新时间
    updated_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))

    @property
    def schema_type(self) -> ApiProviderSchemaType:
        return ApiProviderSchemaType.value_of(self.schema_type_str)
    
    @property
    def tools(self) -> list[ApiToolBundle]:
        return [ApiToolBundle(**tool) for tool in json.loads(self.tools_str)]
    
    @property
    def credentials(self) -> dict:
        return json.loads(self.credentials_str)
    
    @property
    def user(self) -> Account:
        return db.session.query(Account).filter(Account.id == self.user_id).first()

    @property
    def tenant(self) -> Tenant:
        return db.session.query(Tenant).filter(Tenant.id == self.tenant_id).first()

class ToolLabelBinding(db.Model):
    """
    The table stores the labels for tools.
    """
    # 工具标签绑定表
    __tablename__ = 'tool_label_bindings'
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='tool_label_bind_pkey'),
        db.UniqueConstraint('tool_id', 'label_name', name='unique_tool_label_bind'),
    )
    # ID
    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))
    # tool id 工具ID
    tool_id = db.Column(db.String(64), nullable=False)
    # tool type 工具类型
    tool_type = db.Column(db.String(40), nullable=False)
    # label name 标签名称
    label_name = db.Column(db.String(40), nullable=False)

class WorkflowToolProvider(db.Model):
    """
    The table stores the workflow providers.
    """
    # 工具工作流提供商表
    __tablename__ = 'tool_workflow_providers'
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='tool_workflow_provider_pkey'),
        db.UniqueConstraint('name', 'tenant_id', name='unique_workflow_tool_provider'),
        db.UniqueConstraint('tenant_id', 'app_id', name='unique_workflow_tool_provider_app_id'),
    )

    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))
    # name of the workflow provider 工作流提供者名称
    name = db.Column(db.String(40), nullable=False)
    # label of the workflow provider 工作流提供者标签
    label = db.Column(db.String(255), nullable=False, server_default='')
    # icon
    icon = db.Column(db.String(255), nullable=False)
    # app id of the workflow provider 应用ID
    app_id = db.Column(StringUUID, nullable=False)
    # version of the workflow provider 版本
    version = db.Column(db.String(255), nullable=False, server_default='')
    # who created this tool 用户ID
    user_id = db.Column(StringUUID, nullable=False)
    # tenant id  租户ID
    tenant_id = db.Column(StringUUID, nullable=False)
    # description of the provider 描述
    description = db.Column(db.Text, nullable=False)
    # parameter configuration 参数配置
    parameter_configuration = db.Column(db.Text, nullable=False, server_default='[]')
    # privacy policy 隐私政策
    privacy_policy = db.Column(db.String(255), nullable=True, server_default='')
    # 创建时间
    created_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))
    # 更新时间
    updated_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))

    @property
    def schema_type(self) -> ApiProviderSchemaType:
        return ApiProviderSchemaType.value_of(self.schema_type_str)
    
    @property
    def user(self) -> Account:
        return db.session.query(Account).filter(Account.id == self.user_id).first()

    @property
    def tenant(self) -> Tenant:
        return db.session.query(Tenant).filter(Tenant.id == self.tenant_id).first()
    
    @property
    def parameter_configurations(self) -> list[WorkflowToolParameterConfiguration]:
        return [
            WorkflowToolParameterConfiguration(**config)
            for config in json.loads(self.parameter_configuration)
        ]
    
    @property
    def app(self) -> App:
        return db.session.query(App).filter(App.id == self.app_id).first()

class ToolModelInvoke(db.Model):
    """
    store the invoke logs from tool invoke
    """
    __tablename__ = "tool_model_invokes"
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='tool_model_invoke_pkey'),
    )

    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))
    # who invoke this tool
    user_id = db.Column(StringUUID, nullable=False)
    # tenant id
    tenant_id = db.Column(StringUUID, nullable=False)
    # provider 提供者
    provider = db.Column(db.String(40), nullable=False)
    # type 工具类型
    tool_type = db.Column(db.String(40), nullable=False)
    # tool name 工具名称
    tool_name = db.Column(db.String(40), nullable=False)
    # invoke parameters 模型参数
    model_parameters = db.Column(db.Text, nullable=False)
    # prompt messages 提示消息
    prompt_messages = db.Column(db.Text, nullable=False)
    # invoke response 模型响应
    model_response = db.Column(db.Text, nullable=False)
    # 提示令牌
    prompt_tokens = db.Column(db.Integer, nullable=False, server_default=db.text('0'))
    # 回答令牌
    answer_tokens = db.Column(db.Integer, nullable=False, server_default=db.text('0'))
    # 回答单价
    answer_unit_price = db.Column(db.Numeric(10, 4), nullable=False)
    # 回答价格单位
    answer_price_unit = db.Column(db.Numeric(10, 7), nullable=False, server_default=db.text('0.001'))
    # 提供者响应延迟
    provider_response_latency = db.Column(db.Float, nullable=False, server_default=db.text('0'))
    # 总价格
    total_price = db.Column(db.Numeric(10, 7))
    # 货币
    currency = db.Column(db.String(255), nullable=False)
    # 创建时间
    created_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))
    # 更新时间
    updated_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))

class ToolConversationVariables(db.Model):
    """
    store the conversation variables from tool invoke
    """
    __tablename__ = "tool_conversation_variables"
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='tool_conversation_variables_pkey'),
        # add index for user_id and conversation_id
        db.Index('user_id_idx', 'user_id'),
        db.Index('conversation_id_idx', 'conversation_id'),
    )

    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))
    # conversation user id
    user_id = db.Column(StringUUID, nullable=False)
    # tenant id
    tenant_id = db.Column(StringUUID, nullable=False)
    # conversation id  会话ID
    conversation_id = db.Column(StringUUID, nullable=False)
    # variables pool 变量字符串
    variables_str = db.Column(db.Text, nullable=False)

    created_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))
    updated_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))

    @property
    def variables(self) -> dict:
        return json.loads(self.variables_str)
    
class ToolFile(db.Model):
    """
    store the file created by agent
    """
    __tablename__ = "tool_files"
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='tool_file_pkey'),
        # add index for conversation_id
        db.Index('tool_file_conversation_id_idx', 'conversation_id'),
    )

    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))
    # conversation user id
    user_id = db.Column(StringUUID, nullable=False)
    # tenant id
    tenant_id = db.Column(StringUUID, nullable=False)
    # conversation id
    conversation_id = db.Column(StringUUID, nullable=True)
    # file key
    file_key = db.Column(db.String(255), nullable=False)
    # mime type MIME类型
    mimetype = db.Column(db.String(255), nullable=False)
    # original url 原始URL
    original_url = db.Column(db.String(255), nullable=True)