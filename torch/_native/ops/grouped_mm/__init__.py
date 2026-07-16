from .impl import (
    get_use_quack_sm120_grouped_mm,
    register_to_dispatch,
    set_use_quack_sm120_grouped_mm,
    use_quack_sm120_grouped_mm,
)


register_to_dispatch()

__all__ = [
    "get_use_quack_sm120_grouped_mm",
    "set_use_quack_sm120_grouped_mm",
    "use_quack_sm120_grouped_mm",
]
