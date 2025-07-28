from .default import get_default_callbacks

class Callbacks:

    @classmethod
    def add_integration_callbacks(cls, callbacks_list):
        for callbacks in callbacks_list:
            for k, v in callbacks.items():
                if v not in cls._callbacks[k]:
                    cls._callbacks[k].append(v)

    def __init__(self, _callbacks=None, logger=None, logging_config=None):
        if logger is not None:
            self.logger = logger
        else:
            try:
                self.set_logger(
                    log_dir=logging_config.logs_dir,
                    log_stream_level=logging_config.log_stream_level,
                    log_file_level=logging_config.log_file_level,
                )
            except Exception as e:
                print(f"There has been error when creating logger in callbacks: {e}")
                self.logger = None

        self._callbacks = get_default_callbacks()

        if _callbacks is not None:
            self.add_callbacks(_callbacks)

    def add_callback(self, event: str, callback):

        assert (
            event in self._callbacks
        ), f"event '{event}' not found in callbacks {self._callbacks}"
        
        if isinstance(callback, list):
            for _callback in callback:
                assert callable(_callback), RuntimeError(f"{event} must be callable")
                self._callbacks[event].append(_callback)
        else:
            assert callable(callback), RuntimeError(f"{event} must be callable")
            self._callbacks[event].append(callback)

    def add_callbacks(self, callbacks):
        for event, callback in callbacks.items():
            self.add_callback(event, callback)

    def set_callback(self, event: str, callback):

        assert (
            event in self._callbacks
        ), f"event '{event}' not found in callbacks {self._callbacks}"
        assert callable(callback), RuntimeError(f"{event} must be callable")

        self._callbacks[event] = callback

    def run_callbacks(self, event: str, *args, **kwargs):
        for callback in self._callbacks.get(event, []):
            callback(self, *args, **kwargs)