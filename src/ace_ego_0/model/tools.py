"""Small registries used by the public inference package."""


class Registry:
    """Map public framework names to their implementation classes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, type] = {}

    def register(self, key: str):
        """Return a decorator that registers one implementation under ``key``."""

        def decorator(framework_class: type) -> type:
            self._registry[key] = framework_class
            return framework_class

        return decorator

    def __getitem__(self, key: str) -> type:
        """Return the implementation registered under ``key``."""
        return self._registry[key]

    def list(self) -> dict[str, type]:
        """Return all registered implementations."""
        return dict(self._registry)


FRAMEWORK_REGISTRY = Registry("frameworks")
