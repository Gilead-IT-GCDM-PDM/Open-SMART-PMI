"""
Unit tests for helper module
"""
# --- Imports

# Standard library
import unittest

# Local modules
import helper

# --- Tests


class HelperTests(unittest.TestCase):
    """
    Unit tests for helper module
    """

    @staticmethod
    def test_attributes():
        """
        Test for expected functions
        """
        # Methods
        assert callable(helper.readexcel)
        assert callable(helper.compare_output)

    @unittest.skip("Not yet implemented")
    def test_compare_output(self):
        pass

    @unittest.skip("Not yet implemented")
    def test_readexcel(self):
        pass
