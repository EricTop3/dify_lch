
from extensions.ext_database import db
from models import StringUUID
from models.model import Message


class SavedMessage(db.Model):
    # 保存的消息表
    __tablename__ = 'saved_messages'
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='saved_message_pkey'),
        db.Index('saved_message_message_idx', 'app_id', 'message_id', 'created_by_role', 'created_by'),
    )

    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))  #ID
    app_id = db.Column(StringUUID, nullable=False)                            #应用ID
    message_id = db.Column(StringUUID, nullable=False)                        #消息ID
    created_by_role = db.Column(db.String(255), nullable=False, server_default=db.text("'end_user'::character varying"))  #创建者角色
    created_by = db.Column(StringUUID, nullable=False)  #创建者ID
    created_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)')) #创建时间

    @property
    def message(self):
        return db.session.query(Message).filter(Message.id == self.message_id).first()


class PinnedConversation(db.Model):
    # 置顶会话表
    __tablename__ = 'pinned_conversations'
    __table_args__ = (
        db.PrimaryKeyConstraint('id', name='pinned_conversation_pkey'),
        db.Index('pinned_conversation_conversation_idx', 'app_id', 'conversation_id', 'created_by_role', 'created_by'),
    )

    id = db.Column(StringUUID, server_default=db.text('uuid_generate_v4()'))
    app_id = db.Column(StringUUID, nullable=False)
    conversation_id = db.Column(StringUUID, nullable=False)
    created_by_role = db.Column(db.String(255), nullable=False, server_default=db.text("'end_user'::character varying"))
    created_by = db.Column(StringUUID, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP(0)'))
