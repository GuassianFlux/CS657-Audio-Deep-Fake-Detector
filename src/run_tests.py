"""Run all tests."""
import unittest

loader = unittest.TestLoader()
tests = loader.discover('src')
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)