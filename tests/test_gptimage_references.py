#!/usr/bin/env python3
"""Integration test for GPT-Image-1.5 multi-image reference support."""

import asyncio
import os
from pathlib import Path

import pytest

from amplifier_module_image_generation import ImageGenerator


# Create simple test images
def create_test_image(path: Path, color: str) -> None:
    """Create a simple colored PNG for testing."""
    from PIL import Image

    img = Image.new("RGB", (256, 256), color)
    img.save(path)


@pytest.fixture
def test_images(tmp_path: Path) -> list[Path]:
    """Create test reference images."""
    pytest.importorskip("PIL")

    images = []
    colors = ["red", "blue", "green"]

    for idx, color in enumerate(colors):
        img_path = tmp_path / f"ref_{color}.png"
        create_test_image(img_path, color)
        images.append(img_path)

    return images


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
async def test_gptimage_single_reference(test_images: list[Path], tmp_path: Path):
    """Test generation with single reference image."""
    generator = ImageGenerator()

    output_path = tmp_path / "output_single.png"

    result = await generator.generate(
        prompt="Create an abstract pattern inspired by this color",
        output_path=output_path,
        preferred_api="gptimage",
        reference_image_path=test_images[0],
    )

    assert result.success, f"Generation failed: {result.error}"
    assert result.api_used == "gptimage"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"✓ Single reference test passed: {output_path}")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
async def test_gptimage_multiple_references(test_images: list[Path], tmp_path: Path):
    """Test generation with multiple reference images."""
    generator = ImageGenerator()

    output_path = tmp_path / "output_multi.png"

    result = await generator.generate(
        prompt="Combine the colors from these reference images into a gradient",
        output_path=output_path,
        preferred_api="gptimage",
        reference_image_paths=test_images,
    )

    assert result.success, f"Generation failed: {result.error}"
    assert result.api_used == "gptimage"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"✓ Multiple references test passed: {output_path}")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
async def test_gptimage_both_reference_types(test_images: list[Path], tmp_path: Path):
    """Test generation with both single and multiple reference images."""
    generator = ImageGenerator()

    output_path = tmp_path / "output_both.png"

    result = await generator.generate(
        prompt="Use the main reference as the base, incorporating elements from the others",
        output_path=output_path,
        preferred_api="gptimage",
        reference_image_path=test_images[0],
        reference_image_paths=test_images[1:],
    )

    assert result.success, f"Generation failed: {result.error}"
    assert result.api_used == "gptimage"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"✓ Both reference types test passed: {output_path}")


@pytest.mark.asyncio
async def test_gptimage_no_references_still_works(tmp_path: Path):
    """Test that generation without references still works (uses simple API)."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    generator = ImageGenerator()
    output_path = tmp_path / "output_no_ref.png"

    result = await generator.generate(
        prompt="A simple geometric pattern",
        output_path=output_path,
        preferred_api="gptimage",
    )

    assert result.success, f"Generation failed: {result.error}"
    assert result.api_used == "gptimage"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"✓ No references test passed: {output_path}")


if __name__ == "__main__":
    # Run tests manually for debugging
    import sys

    async def run_manual_tests():
        tmp = Path("/tmp/gptimage_test")
        tmp.mkdir(exist_ok=True)

        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY not set - skipping API tests")
            print("Set OPENAI_API_KEY to test actual generation")
            sys.exit(0)

        # Import PIL here for manual testing
        try:
            from PIL import Image
        except ImportError:
            print("⚠️  Pillow not installed - installing...")
            os.system("pip install pillow")
            from PIL import Image

        # Create test images
        print("Creating test images...")
        test_imgs = []
        for color in ["red", "blue", "green"]:
            img_path = tmp / f"ref_{color}.png"
            create_test_image(img_path, color)
            test_imgs.append(img_path)
            print(f"  Created: {img_path}")

        print("\nTest 1: Single reference image")
        try:
            await test_gptimage_single_reference(test_imgs, tmp)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        print("\nTest 2: Multiple reference images")
        try:
            await test_gptimage_multiple_references(test_imgs, tmp)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        print("\nTest 3: Both reference types")
        try:
            await test_gptimage_both_reference_types(test_imgs, tmp)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        print("\nTest 4: No references (simple API)")
        try:
            await test_gptimage_no_references_still_works(tmp)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print(f"Generated images: {tmp}")
        return True

    success = asyncio.run(run_manual_tests())
    sys.exit(0 if success else 1)
