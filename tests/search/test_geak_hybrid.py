"""Tests for geak_hybrid search algorithm registration and config."""

import pytest


class TestGEAKHybridRegistration:
    """Verify geak_hybrid is properly registered in SkyDiscover."""

    def test_config_type_mapping(self):
        """GEAKHybridDatabaseConfig should be in _DB_CONFIG_BY_TYPE."""
        from skydiscover.config import GEAKHybridDatabaseConfig, _DB_CONFIG_BY_TYPE

        assert "geak_hybrid" in _DB_CONFIG_BY_TYPE
        assert _DB_CONFIG_BY_TYPE["geak_hybrid"] is GEAKHybridDatabaseConfig

    def test_config_inherits_adaevolve(self):
        """GEAKHybridDatabaseConfig should extend AdaEvolveDatabaseConfig."""
        from skydiscover.config import AdaEvolveDatabaseConfig, GEAKHybridDatabaseConfig

        assert issubclass(GEAKHybridDatabaseConfig, AdaEvolveDatabaseConfig)

    def test_config_geak_fields(self):
        """GEAKHybridDatabaseConfig should have GEAK-specific fields."""
        from skydiscover.config import GEAKHybridDatabaseConfig

        cfg = GEAKHybridDatabaseConfig()
        assert cfg.inner_loop_budget == 3
        assert cfg.max_invalid_rounds == 2
        assert cfg.accept_improvement_epsilon == 1e-6

    def test_cli_search_choices(self):
        """geak_hybrid should be in CLI search choices."""
        from skydiscover.cli import _SEARCH_CHOICES

        assert "geak_hybrid" in _SEARCH_CHOICES

    def test_controller_registry(self):
        """geak_hybrid should be in the controller registry."""
        from skydiscover.search.registry import _CONTROLLER_REGISTRY

        # Force route imports to trigger registration
        import skydiscover.search.route  # noqa: F401

        assert "geak_hybrid" in _CONTROLLER_REGISTRY

    def test_database_registry(self):
        """geak_hybrid should be in the database registry."""
        from skydiscover.search.registry import _DATABASE_REGISTRY

        import skydiscover.search.route  # noqa: F401

        assert "geak_hybrid" in _DATABASE_REGISTRY

    def test_controller_class(self):
        """GEAKHybridController should extend AdaEvolveController."""
        from skydiscover.search.adaevolve.controller import AdaEvolveController
        from skydiscover.search.geak_hybrid.controller import GEAKHybridController

        assert issubclass(GEAKHybridController, AdaEvolveController)

    def test_config_from_yaml_dict(self):
        """Config.from_dict with search.type=geak_hybrid should parse GEAK fields."""
        from skydiscover.config import Config

        config_dict = {
            "search": {
                "type": "geak_hybrid",
                "database": {
                    "inner_loop_budget": 5,
                    "max_invalid_rounds": 3,
                    "num_islands": 2,
                },
            },
        }
        config = Config.from_dict(config_dict)
        assert config.search.type == "geak_hybrid"
        assert config.search.database.inner_loop_budget == 5
        assert config.search.database.max_invalid_rounds == 3
        assert config.search.database.num_islands == 2  # inherited from AdaEvolve
