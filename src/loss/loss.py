class Loss:
    """
    Base class for loss function.
    """

    def apply(self, *args, **kwargs):
        raise NotImplementedError()
