"""Spert model

.. autosummary::
    :toctree:

    conll04_constants
    conll04_input_generator
    internal_constants
    internal_input_generator
    model
    scierc_constants
    scierc_input_generator
"""


class SpertConfig:
    """

    """
    def __init__(self, dataset):
        if dataset is None:
            raise ValueError("Dataset argument not found")
        elif dataset == "conll04":
            import renard_joint.spert.conll04_constants as constants
            import renard_joint.spert.conll04_input_generator as input_generator
        elif dataset == "scierc":
            import renard_joint.spert.scierc_constants as constants
            import renard_joint.spert.scierc_input_generator as input_generator
        elif dataset == "internal":
            import renard_joint.spert.internal_constants as constants
            import renard_joint.spert.internal_input_generator as input_generator
        else:
            raise ValueError("Invalid dataset argument")

        self.constants = constants
        self.input_generator = input_generator
