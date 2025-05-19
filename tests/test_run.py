import importlib
import sys
import types
import unittest

# create stub modules for missing dependencies before importing run
sys.modules['numpy'] = types.ModuleType('numpy')
sys.modules['cv2'] = types.ModuleType('cv2')
sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')
scipy_stats = types.ModuleType('scipy.stats')
scipy_stats.chisquare = lambda f_obs, f_exp: (0, 1)
sys.modules['scipy'] = types.ModuleType('scipy')
sys.modules['scipy.stats'] = scipy_stats
PIL = types.ModuleType('PIL')
PIL.ImageColor = types.SimpleNamespace(getrgb=lambda c: (18, 52, 86))
sys.modules['PIL'] = PIL

import run
importlib.reload(run)

class TestHexToRgb(unittest.TestCase):
    def test_hex_to_rgb(self):
        self.assertEqual(run.hex_to_rgb('#123456'), (18, 52, 86))

if __name__ == '__main__':
    unittest.main()
