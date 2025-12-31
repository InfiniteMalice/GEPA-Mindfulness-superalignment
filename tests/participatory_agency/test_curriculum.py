from __future__ import annotations

from gepa_mindfulness.participatory_agency.training.curriculum import (
    DEFAULT_CURRICULUM,
    CurriculumPhase,
    get_default_curriculum,
)


def test_default_curriculum_phases() -> None:
    curriculum = get_default_curriculum()
    assert len(curriculum) == 5
    assert isinstance(curriculum[0], CurriculumPhase)
    assert curriculum[0].active_heads == ("epistemic",)
    assert curriculum[-1].name == "phase_5_integrated"


def test_curriculum_is_ordered() -> None:
    names = [phase.name for phase in DEFAULT_CURRICULUM]
    assert names == [
        "phase_1_epistemic",
        "phase_2_cooperation",
        "phase_3_flexibility",
        "phase_4_belonging",
        "phase_5_integrated",
    ]
