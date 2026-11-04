"""Smoke test suite to verify scaffolding integrity and module discoverability."""

from pathlib import Path


def test_required_scaffolding_files_exist(project_root: Path):
    """Verify all top-level configuration and documentation files exist."""
    required_files = [
        "pyproject.toml",
        "docker-compose.yml",
        ".env.example",
        ".gitignore",
        "README.md",
    ]
    for rel_path in required_files:
        file_path = project_root / rel_path
        assert file_path.is_file(), f"Missing required file: {rel_path}"


def test_required_module_directories_exist(project_root: Path):
    """Verify all core module directories exist."""
    required_dirs = [
        "agent",
        "agent/nodes",
        "agent/prompts",
        "mcp_server",
        "mcp_server/tools",
        "rag",
        "finetuning",
        "finetuning/configs",
        "api",
        "api/routes",
        "eval",
        "web",
        "tests",
    ]
    for rel_path in required_dirs:
        dir_path = project_root / rel_path
        assert dir_path.is_dir(), f"Missing required directory: {rel_path}"


def test_python_packages_importable():
    """Verify that all core Python modules can be imported without syntax errors."""
    import agent
    import agent.state

    assert hasattr(agent.state, "ResearchState")


def test_research_state_schema():
    """Verify that ResearchState Pydantic schema matches the PRD specification."""
    from agent.state import ResearchState

    state = ResearchState(research_topic="Test Topic")
    assert state.research_topic == "Test Topic"
    assert state.outline == []
    assert state.search_queries == []
    assert state.scraped_urls == set()
    assert state.evidence_chunks == []
    assert state.sections == {}
    assert state.unresolved_gaps == []
    assert state.iteration_count == 0
    assert state.max_iterations == 2
    assert state.final_report is None
