"""Canonical workflow stage ordering."""

from enum import Enum


class Stage(str, Enum):
    PREPROCESS = "preprocess"
    EOF = "eof"
    TRAIN_LIM = "train-lim"
    CALIBRATE_PAGES2K = "calibrate-pages2k"
    PREPARE_TEMP12K = "prepare-temp12k"
    ESTIMATE_BOREHOLE_ERROR = "estimate-borehole-error"
    ASSIMILATE = "assimilate"
    RECONSTRUCT = "reconstruct"
    VERIFY = "verify"

    @classmethod
    def ordered(cls) -> tuple["Stage", ...]:
        return tuple(cls)
