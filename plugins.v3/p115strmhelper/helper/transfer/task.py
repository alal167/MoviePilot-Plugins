import time
import uuid
from threading import Lock, Timer
from typing import Callable, Dict, List, Optional, Set, Tuple

from app.log import logger

from ...schemas.transfer import TransferTask


class TransferTaskManager:
    """
    整理任务队列管理器

    幂等规则：同一「源文件 -> 目标文件」在待处理队列中只保留一个任务，
    批次处理完成后会记住该任务一段时间（settled_ttl），在窗口内重复投递
    的同一任务直接忽略，避免上游重复触发导致的重复整理。
    """

    def __init__(
        self,
        batch_delay: float = 10.0,
        batch_max_size: int = 100,
        batch_callback: Optional[Callable[[List[TransferTask]], None]] = None,
        settled_ttl: float = 1800.0,
        failed_ttl: float = 86400.0,
    ):
        """
        初始化任务管理器

        :param batch_delay (float): 批量等待时间（秒），默认 10.0 秒
        :param batch_max_size (int): 单批次最大任务数，默认 100
        :param batch_callback (Callable): 批量处理回调函数，接收任务列表作为参数
        :param settled_ttl (float): 已处理任务幂等窗口（秒），默认 1800 秒
        :param failed_ttl (float): 终态失败任务幂等窗口（秒），默认 86400 秒
        """
        self.batch_delay = batch_delay
        self.batch_max_size = batch_max_size
        self.batch_callback = batch_callback
        self.settled_ttl = settled_ttl
        self.failed_ttl = failed_ttl

        # 待处理任务队列
        self._pending_tasks: List[TransferTask] = []

        # 待处理任务的幂等键集合
        self._pending_keys: Set[str] = set()

        # 已完成/已失败任务的幂等记录 {幂等键: (记录时间, 生效时长)}
        self._settled_keys: Dict[str, Tuple[float, float]] = {}

        # 线程锁，保护共享资源
        self._lock = Lock()

        # 延迟定时器
        self._timer: Optional[Timer] = None

        # 是否正在处理批量任务
        self._processing = False

        logger.info(
            f"【整理接管】初始化完成，批量延迟: {batch_delay} 秒，最大批次: {batch_max_size}，"
            f"幂等窗口: {settled_ttl} 秒，失败隔离: {failed_ttl} 秒"
        )

    @staticmethod
    def _normalize_path(value) -> str:
        """
        归一化路径字符串，用于生成稳定的幂等键
        """
        if value is None:
            return ""
        text = str(value).strip().replace("\\", "/")
        while len(text) > 1 and text.endswith("/"):
            text = text[:-1]
        return text.lower()

    @classmethod
    def _build_task_key(cls, task: TransferTask) -> Optional[str]:
        """
        生成任务幂等键：源文件路径 + 目标文件路径

        :param task (TransferTask): 整理任务
        :return: 幂等键，无法生成时返回 None（此时不做幂等控制）
        """
        try:
            source = cls._normalize_path(task.fileitem.path if task.fileitem else None)
            target = cls._normalize_path(task.target_path)
        except Exception:
            return None
        if not source and not target:
            return None
        return f"{source}|{target}"

    @staticmethod
    def _task_label(task: TransferTask) -> str:
        """
        任务描述，用于日志输出
        """
        try:
            name = task.fileitem.name if task.fileitem else ""
        except Exception:
            name = ""
        return f"{name} -> {task.target_path}"

    def _purge_expired_locked(self) -> None:
        """
        清理过期的幂等记录（需在锁内调用）
        """
        if not self._settled_keys:
            return
        now = time.time()
        expired = [
            key
            for key, (record_time, ttl) in self._settled_keys.items()
            if now - record_time >= ttl
        ]
        for key in expired:
            self._settled_keys.pop(key, None)

    def add_task(self, task: TransferTask) -> bool:
        """
        添加任务到待处理队列

        :param task (TransferTask): 整理任务
        :return: 是否真正加入队列，False 表示被幂等拦截
        """
        should_trigger_immediately = False
        task_key = self._build_task_key(task)

        with self._lock:
            if task_key is not None:
                self._purge_expired_locked()
                if task_key in self._pending_keys:
                    logger.info(
                        f"【整理接管】忽略重复任务（已在待处理队列）: {self._task_label(task)}"
                    )
                    return False
                if task_key in self._settled_keys:
                    logger.info(
                        f"【整理接管】忽略重复任务（已处理完成或已失败，不再重复整理）: "
                        f"{self._task_label(task)}"
                    )
                    return False

            # 检查是否达到最大批次大小
            if len(self._pending_tasks) >= self.batch_max_size:
                logger.warn(
                    f"【整理接管】待处理队列已满（{self.batch_max_size}），"
                    f"立即触发批量处理"
                )
                should_trigger_immediately = True

            # 添加任务到队列
            self._pending_tasks.append(task)
            if task_key is not None:
                self._pending_keys.add(task_key)
            logger.debug(
                f"【整理接管】任务已加入队列: {task.fileitem.name} -> {task.target_path}，"
                f"当前队列大小: {len(self._pending_tasks)}"
            )

            # 如果达到最大批次大小，立即触发批量处理
            if should_trigger_immediately:
                # 取消现有定时器
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None
            else:
                # 重置延迟定时器
                self._reset_timer()

        # 在锁外触发批量处理
        if should_trigger_immediately:
            self._trigger_batch_process()

        return True

    def _reset_timer(self) -> None:
        """
        重置延迟定时器
        每次新任务到达时调用，延迟 batch_delay 秒后触发批量处理
        """
        # 取消现有定时器
        if self._timer is not None:
            self._timer.cancel()

        # 创建新的定时器
        self._timer = Timer(
            interval=self.batch_delay, function=self._trigger_batch_process
        )
        self._timer.daemon = True
        self._timer.start()

        logger.debug(
            f"【整理接管】延迟定时器已重置，将在 {self.batch_delay} 秒后触发批量处理"
        )

    def _settle_tasks(
        self, tasks: List[TransferTask], ttl: Optional[float] = None
    ) -> None:
        """
        将批次任务登记为已处理，在幂等窗口内忽略重复投递

        :param tasks (List): 已处理的任务列表
        :param ttl (float): 生效时长，默认使用 settled_ttl
        """
        expire = self.settled_ttl if ttl is None else ttl
        if not tasks:
            return
        now = time.time()
        with self._lock:
            self._purge_expired_locked()
            for task in tasks:
                task_key = self._build_task_key(task)
                if task_key is None:
                    continue
                # 处理中的幂等键保留到批次结算后再释放，
                # 避免处理期间上游重复投递被再次入队
                self._pending_keys.discard(task_key)
                # 已有的终态失败记录优先保留，避免被较短窗口覆盖
                if task_key in self._settled_keys:
                    continue
                self._settled_keys[task_key] = (now, expire)

    def mark_terminal_failure(self, task: TransferTask) -> None:
        """
        标记任务为终态失败（不可自动重试），在失败隔离窗口内不再重复整理

        :param task (TransferTask): 失败任务
        """
        task_key = self._build_task_key(task)
        if task_key is None:
            return
        with self._lock:
            self._purge_expired_locked()
            self._pending_keys.discard(task_key)
            self._settled_keys[task_key] = (time.time(), self.failed_ttl)
        logger.warn(
            f"【整理接管】任务已标记为终态失败，{int(self.failed_ttl)} 秒内不再重复整理: "
            f"{self._task_label(task)}"
        )

    def _trigger_batch_process(self) -> None:
        """
        触发批量处理
        从待处理队列中取出所有任务，调用批量处理回调
        """
        with self._lock:
            # 如果正在处理，跳过
            if self._processing:
                logger.debug("【整理接管】批量处理正在进行中，跳过本次触发")
                return

            # 如果没有待处理任务，直接返回
            if not self._pending_tasks:
                logger.debug("【整理接管】没有待处理任务，跳过批量处理")
                return

            # 取出所有待处理任务
            tasks_to_process = self._pending_tasks.copy()
            self._pending_tasks.clear()

            # 取消定时器
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            # 标记为正在处理
            self._processing = True

            transfer_batch_id = uuid.uuid4().hex
            for task in tasks_to_process:
                task.transfer_batch_id = transfer_batch_id

        # 在锁外执行批量处理，避免阻塞
        try:
            task_count = len(tasks_to_process)
            logger.info(f"【整理接管】开始批量处理 {task_count} 个任务")

            if self.batch_callback:
                self.batch_callback(tasks_to_process)
            else:
                logger.warn("【整理接管】未设置批量处理回调函数，任务将被丢弃")

            logger.info(f"【整理接管】批量处理完成，共处理 {task_count} 个任务")
        except Exception as e:
            logger.error(f"【整理接管】批量处理异常: {e}", exc_info=True)
        finally:
            # 批次处理结束后登记幂等，成功与失败任务都不再重复整理
            self._settle_tasks(tasks_to_process)

            # 重置处理标志
            with self._lock:
                self._processing = False
                pending_count = len(self._pending_tasks)
                should_trigger_next = pending_count > 0
                if should_trigger_next:
                    # 新任务通过正常延迟触发，避免处理完成后的紧密递归循环
                    self._reset_timer()

            if should_trigger_next:
                logger.info(
                    f"【整理接管】批量处理完成后，队列中仍有 {pending_count} 个待处理任务，"
                    f"将在 {self.batch_delay} 秒后处理"
                )

    def flush(self) -> None:
        """
        立即处理所有待处理任务（不等待延迟）
        用于插件关闭或手动触发时
        """
        logger.info("【整理接管】手动触发批量处理（flush）")
        self._trigger_batch_process()

    def get_pending_count(self) -> int:
        """
        获取待处理任务数量

        :return: 待处理任务数量
        """
        with self._lock:
            return len(self._pending_tasks)

    def shutdown(self) -> None:
        """
        关闭任务管理器
        取消定时器并丢弃剩余任务（不再触发整理，避免关闭流程重复执行）
        """
        logger.info("【整理接管】正在关闭...")

        with self._lock:
            # 取消定时器
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            pending_count = len(self._pending_tasks)
            for task in self._pending_tasks:
                task_key = self._build_task_key(task)
                if task_key is not None:
                    self._pending_keys.discard(task_key)
            self._pending_tasks.clear()

        if pending_count > 0:
            logger.info(
                f"【整理接管】关闭时丢弃 {pending_count} 个未处理任务，不再触发整理"
            )

        logger.info("【整理接管】已关闭")
