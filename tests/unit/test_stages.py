from lmr4dvar.workflow import Stage


def test_stage_order() -> None:
    assert [stage.value for stage in Stage.ordered()] == [
        "preprocess",
        "eof",
        "train-lim",
        "calibrate-pages2k",
        "prepare-temp12k",
        "estimate-borehole-error",
        "assimilate",
        "reconstruct",
        "verify",
    ]
