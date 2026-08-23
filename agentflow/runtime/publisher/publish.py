import logging

from injectq import Inject, InjectQ, inject

from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.runtime.publisher.events import EventModel
from agentflow.utils.background_task_manager import BackgroundTaskManager


logger = logging.getLogger("agentflow.publisher")


async def _publish_event_task(
    event: EventModel,
    publisher: BasePublisher | None,
) -> None:
    """Publish an event asynchronously if publisher is configured.

    Args:
        event: The event to publish.
        publisher: The publisher instance, or None.
    """
    if publisher:
        try:
            await publisher.publish(event)
            logger.debug("Published event: %s", event)
        except Exception as e:
            logger.error("Failed to publish event: %s", e)


@inject
def publish_event(
    event: EventModel,
    publisher: BasePublisher | None = Inject[BasePublisher],
    task_manager: BackgroundTaskManager | None = Inject[BackgroundTaskManager],
) -> None:
    """Publish an event asynchronously using the background task manager.

    Args:
        event: The event to publish.
        publisher: The publisher instance (injected).
        task_manager: The background task manager (injected).
    """
    if publisher is None:
        publisher = InjectQ.get_instance().try_get(BasePublisher)
    if publisher is None:
        return

    if task_manager is None:
        task_manager = InjectQ.get_instance().try_get(BackgroundTaskManager)
    if task_manager is None:
        return

    # Snapshot the event so in-place mutations by the caller (e.g. changing event_type
    # from START to END on the same object) do not corrupt the asynchronous task.
    event_copy = event.model_copy(deep=True)
    task_manager.create_task(_publish_event_task(event_copy, publisher))
