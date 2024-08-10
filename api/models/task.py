from datetime import datetime, timezone

from celery import states

from extensions.ext_database import db


class CeleryTask(db.Model):
    """Task result/status."""
    # Celery 任务元数据表
    __tablename__ = 'celery_taskmeta'
    # 自增主键，任务ID序列
    id = db.Column(db.Integer, db.Sequence('task_id_sequence'),
                   primary_key=True, autoincrement=True)
    # 任务ID 唯一任务标识
    task_id = db.Column(db.String(155), unique=True)
    # 状态 默认值为 PENDING
    status = db.Column(db.String(50), default=states.PENDING)
    # 任务结果，使用 PickleType 存储 
    result = db.Column(db.PickleType, nullable=True)
    # 任务完成时间，默认当前 UTC 时间，更新时也会记录当前 UTC 时间
    date_done = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
                          onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=True)
    # 任务出错时的回溯信息
    traceback = db.Column(db.Text, nullable=True)
    name = db.Column(db.String(155), nullable=True)
    # 任务的参数，以二进制形式存储
    args = db.Column(db.LargeBinary, nullable=True)
    # 任务的关键字参数，以二进制形式存储
    kwargs = db.Column(db.LargeBinary, nullable=True)
    # 执行任务的工人
    worker = db.Column(db.String(155), nullable=True)
    # 任务重试次数
    retries = db.Column(db.Integer, nullable=True)
    # 任务所在的队列
    queue = db.Column(db.String(155), nullable=True)


class CeleryTaskSet(db.Model):
    """TaskSet result."""
    # Celery 任务集合元数据表
    __tablename__ = 'celery_tasksetmeta'
    # id: 自增主键，唯一标识任务集记录
    id = db.Column(db.Integer, db.Sequence('taskset_id_sequence'),
                   autoincrement=True, primary_key=True)
    # taskset_id: 任务集的唯一标识符
    taskset_id = db.Column(db.String(155), unique=True)
    # 任务集结果，使用 PickleType 存储
    result = db.Column(db.PickleType, nullable=True)
    #  任务集完成时间，默认当前 UTC 时间
    date_done = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
                          nullable=True)
