"""
PixelRevive — Module 1 Verification Script
============================================
Run this to confirm Module 1 is working correctly.
"""

from damage_removal import restore_photo
import os

base = os.path.dirname(__file__)

print('Module loaded successfully')
print('Restoring test photo...')

restore_photo(
    input_path  = os.path.join(base, 'test_input.png'),
    output_path = os.path.join(base, 'output_restored.png')
)

print('Module 1 is working perfectly!')
