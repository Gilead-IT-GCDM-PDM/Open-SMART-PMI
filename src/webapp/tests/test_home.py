"""
Unit tests for ActiveUsers class
"""
# --- Imports

# Standard library
import os
import unittest

# External packages
# import pandas as pd
# import pytest

# Local modules
from reconcile import Reconcile


_MISSING_EXCEPTION_MSG = 'Expected missing column exception. Msgs = %s'
_DUP_EXCEPTION_MSGS = 'Curricula Report: Dup row exception expected:  %s'


class ReconcileTest(unittest.TestCase):
    """
    Regression test for Reconcilation class

    GROUP_NAME, USER_NAME, FULL_NAME, USER_ROLE, GxP Role, Active User,
        GxP qualified, Code, Error_Code
    """

    @classmethod
    def setUpClass(cls):  # pylint: disable=invalid-name
        """
        Perform preparations required by most tests.
        """

        data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                'data', 'Req Id 16', 'Scenario 1',
                                'Input Files')

        cls.rec = Reconcile.from_directory(data_dir)

    @classmethod
    def tearDownClass(cls):  # pylint: disable=invalid-name
        """
        Clean up after all the tests.
        """
        pass

    @staticmethod
    def test_attributes():
        """
        Test for expected attributes
        """
        # Properties
        assert hasattr(Reconcile, 'all_users')
        assert hasattr(Reconcile, 'exceptions')

    def test_assigned_curriculum_roles(self):
        """
        Tests role associated with a given curriculum
        """
        user_id1 = 'JABDELFATTAH'
        actual1 = self.rec.assigned_curriculum_roles(user_id1)
        expected1 = {'SD_EM_ANALYST'}

        assert actual1 == expected1, \
               'Role: %s = %s, Expected: %s' % (user_id1, actual1, expected1)

    @unittest.skip("Not yet implemented")
    def test_reconcile_user_role(self):
        pass

    @unittest.skip("Not yet implemented")
    def is_gxp_not_qualified(self):
        pass

    @unittest.skip("Not yet implemented")
    def test_reconcile_user(self):
        pass

    @unittest.skip("Not yet implemented")
    def test_fuzzy_output(self):
        pass

    @unittest.skip("Not yet implemented")
    def test_fuzzy_summary(self):
        pass

    @staticmethod
    def test_missing_column_exception():
        data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                'data', 'exceptions/Missing column')
        rec = Reconcile.from_directory(data_dir)
        msgs = rec.exceptions

        gxp_exc = "Curricula Report: Missing column: {'GXPLearn Curriculum'}"
        active_users_exc = "Active Users: Missing column: {'FULL_NAME'}"

        # check if expected msgs are there
        assert gxp_exc in msgs, _MISSING_EXCEPTION_MSG % msgs
        assert active_users_exc in msgs,  _MISSING_EXCEPTION_MSG % msgs

    @staticmethod
    def test_incorrect_header_exception():
        data_dir = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'data', 'exceptions/Incorrect header')
        rec = Reconcile.from_directory(data_dir)

        assert len(rec.exceptions) > 0 and 'None' not in rec.exceptions, \
            'Exception expected for invalid GxPLearnFile'

    @staticmethod
    def test_duplicate_row_exception():
        data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                'data', 'exceptions/Duplicate rows')
        rec = Reconcile.from_directory(data_dir)
        msgs = rec.exceptions

        dup_exc = 'Curricula Report: RowIds duplicated: [40, 43]'
        # check if expected msgs are there
        assert dup_exc in msgs, _DUP_EXCEPTION_MSGS % msgs

    @staticmethod
    def test_location_for_inactive_user():
        data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                'data', 'exceptions/Inactive user')
        rec = Reconcile.from_directory(data_dir)
        loc_agalaz = rec.location('AGALAZ', 'VIEW_ONLY')
        loc_acurley = rec.location('ACURLEY', 'LAB_MANAGER')
        loc_bdaly = rec.location('BDALY', 'EDM_LAB_ANALYST')
        loc_wtsai = rec.location('WTSAI', 'OC_MGF')

        assert loc_agalaz == 'FC', 'Expected FC found: %s' % loc_agalaz
        assert loc_acurley == 'FC,GSIUC', 'Expected FC,GSIUC found: %s' % loc_acurley
        assert loc_bdaly == 'EDM', 'Expected empty found: %s' % loc_bdaly
        assert loc_wtsai == 'OC', 'Expected empty found: %s' % loc_wtsai


    def test_full_name(self):
        """
        Tests role associated with a given curriculum
        """
        user_id1 = 'JABDELFATTAH'
        actual1 = self.rec.full_name(user_id1)
        expected1 = 'Joseph Abdelfattah'

        user_id2 = 'UserIdNotPresent'
        actual2 = self.rec.full_name(user_id2)
        expected2 = 'UserIdNotPresent'

        assert actual1 == expected1, \
            'UserId: %s = %s, Expected: %s' % (user_id1, actual1, expected1)
        assert actual2 == expected2, \
            'UserId: %s = %s, Expected: %s' % (user_id2, actual2, expected2)
