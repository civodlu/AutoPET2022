import unittest
import trw.train
from trw.callbacks.callback import Callback


class CallbackRecordEpoch(Callback):
    def __init__(self):
        self.epochs_called = []

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        self.epochs_called.append(len(history))


class TestCallbackSkipEpoch(unittest.TestCase):
    def test_basic(self):
        callback_record = CallbackRecordEpoch()
        callback = trw.callbacks.CallbackSkipEpoch(nb_epochs=3, callbacks=[callback_record])

        nb_epochs = 200
        for h in range(nb_epochs):
            callback(None, [1] * h, None, None, None, None, None, None)

        assert len(callback_record.epochs_called) == nb_epochs // 3 + 1
        for e in callback_record.epochs_called:
            assert e % 3 == 0


if __name__ == '__main__':
    unittest.main()
