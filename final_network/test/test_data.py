"""
test_data.py - Test cases for the grabber and batcher
author : Benjamin Blundell
email : me@benjamin.computer

https://docs.python.org/3/library/unittest.html

"""

import unittest, os

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)
from common import gen_data

class TestGenPostData(unittest.TestCase):
  ''' Test the Postgres database. We assume one is running locally.''' 
  def setUp(self):
    self.grabber = gen_data.Grabber()

  def test_grab(self):
    (loops,summary) = self.grabber.grab(limit=1)
    self.assertEqual(len(loops),1,"Limit of 1 does not match returned size")

  def tearDown(self):
    pass

def suite():
  suite = unittest.TestSuite()
  suite.addTest(TestGenPostData('test_grab'))
  return suite

if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  runner.run(suite())

