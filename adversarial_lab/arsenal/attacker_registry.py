


class AttackerRegistry:
    _registry = {
        "adversarial": {
            "blackbox": {},
            "whitebox": {},
        },

    }

    @classmethod
    def get_all_attacks(cls) -> dict:
        return cls._registry