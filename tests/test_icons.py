import unittest
import xml.etree.ElementTree as ET

from pathsim.blocks import Block

import pathsim_fusion


class TestIcons(unittest.TestCase):
    """Every block of the toolbox has an icon, read by Block.icon()."""

    def test_blocks_have_icons(self):
        blocks = [
            getattr(pathsim_fusion, name)
            for name in pathsim_fusion.__all__
            if isinstance(getattr(pathsim_fusion, name), type)
            and issubclass(getattr(pathsim_fusion, name), Block)
        ]
        self.assertGreater(len(blocks), 0)

        for cls in blocks:
            with self.subTest(block=cls.__name__):
                icon = cls.icon()
                self.assertIsNotNone(icon)
                root = ET.fromstring(icon)
                self.assertEqual(root.get("viewBox"), "0 0 96 64")


if __name__ == "__main__":
    unittest.main()
