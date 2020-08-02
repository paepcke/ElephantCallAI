'''
Created on Aug 2, 2020

@author: paepcke
'''
import os
from pathlib import Path
import sys
import tempfile
import shutil
import unittest


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dsp_utils import DSPUtils

TEST_ALL = True
#TEST_ALL = False

class TestDSPUtils(unittest.TestCase):

    #------------------------------------
    # setUpClass 
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        cls.curr_dir = os.path.dirname(__file__)

    #------------------------------------
    # setUp 
    #-------------------

    def setUp(self):
        pass

    #------------------------------------
    # tearDown 
    #-------------------

    def tearDown(self):
        try:
            shutil.rmtree(str(self.tmp_dir_obj))
        except Exception:
            # Best effort:
            pass

    #------------------------------------
    # testFind 
    #-------------------

    
    def testFind(self):
        
        # Create a test directory structure:
        
        # ---TmpDir Below Test Dir
        #   --- level0 
        #          --- no_match0_1.foo
        #          --- match0_1.txt
        #   --- level1
        #   --- level2
        #          --- no_match2_1.foo
        #          --- match2_1.txt
        #   --- level3
        #          --- match3_1.txt
        #          --- match3_2.txt
       
        self.tmp_dir_obj = tempfile.TemporaryDirectory(prefix='find_cmd_tmp', 
                                                       dir=self.curr_dir)
        tmp_dir = self.tmp_dir_obj.name
        os.makedirs(os.path.join(tmp_dir,'level0/level1/level2/level3'))
        lev0 = Path(os.path.join(tmp_dir,'level0'))
        _lev1 = Path(os.path.join(tmp_dir,'level0/level1'))
        lev2 = Path(os.path.join(tmp_dir,'level0/level1/level2'))
        lev3 = Path(os.path.join(tmp_dir,'level0/level1/level2/level3'))
                    
        Path(lev0).joinpath('no_match0_1.foo').touch()
        
        match0_1_path = Path(lev0).joinpath('match0_1.txt') 
        match0_1_path.touch()

        Path(lev2).joinpath('no_match2_1.foo').touch()

        match2_1_path = Path(lev2).joinpath('match2_1.txt')
        match2_1_path.touch()
        
        match3_1_path = Path(lev3).joinpath('match3_1.txt')
        match3_1_path.touch()
        
        match3_2_path = Path(lev3).joinpath('match3_2.txt')
        match3_2_path.touch()
        
        full_path_set = set(DSPUtils.unix_find(lev0, r'.*\.txt'))
        self.assertSetEqual(full_path_set, 
                          set([str(match0_1_path),
                              str(match2_1_path),
                              str(match3_1_path),
                              str(match3_2_path)
                              ]))

# --------------- Main -----------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()