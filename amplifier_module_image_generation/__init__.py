"""Multi-provider AI image generation for Amplifier applications.

This module provides both a standalone library interface and an Amplifier Tool protocol interface.

Library Usage (Direct Import):
    >>> from amplifier_module_image_generation import ImageGenerator
    >>> generator = ImageGenerator()
    >>> result = await generator.generate(
    ...     prompt="A serene landscape",
    ...     output_path=Path("output/image.png")
    ... )

Tool Usage (Via Amplifier):
    >>> from amplifier_module_image_generation import ImageGenerationTool
    >>> tool = ImageGenerationTool()
    >>> result = await tool.execute({
    ...     "operation": "generate",
    ...     "prompt": "A serene landscape",
    ...     "output_path": "output/image.png"
    ... })
"""

# Amplifier module metadata
__amplifier_module_type__ = "tool"

import logging
from typing import Any

from amplifier_core import ModuleCoordinator

from .generator import ImageGenerator
from .models import ImageGenerationError, ImageResult
from .tool import ImageGenerationTool

__version__ = "0.1.0"
__all__ = [
    "ImageGenerator",
    "ImageResult",
    "ImageGenerationError",
    "ImageGenerationTool",
    "mount",
]

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None) -> None:
    """Mount image-generation tool.

    Args:
        coordinator: Module coordinator for registering tools
        config: Optional configuration

    Returns:
        None

    Raises:
        ImportError: If required packages are not installed
    """
    # Create and register tool
    tool = ImageGenerationTool()
    await coordinator.mount("tools", tool, name=tool.name)

    logger.info("Mounted image-generation tool")
