"""Cooperative cancellation at model and persistence boundaries."""
from contextlib import contextmanager


class OperationCancelled(Exception):
    pass


def check_stop(event):
    if event is not None and event.is_set():
        raise OperationCancelled('Stopped.')


@contextmanager
def publish_operation(operation, lock):
    # Stop and publication share only this short critical section, never inference.
    with lock:
        check_stop(operation['stop'])
        yield
        operation['complete'] = True
