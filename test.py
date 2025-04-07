import numpy as np
import unittest
from parser import rearrange
from einopserr import EinopsError

class TestRearrange(unittest.TestCase):
    """Test cases for the rearrange function."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_identity(self):
        """Test identity transformation (no change)."""
        x = np.random.rand(2, 3, 4)
        result = rearrange(x, 'a b c -> a b c')
        np.testing.assert_array_equal(x, result)
    
    def test_transpose(self):
        """Test simple transposition."""
        x = np.random.rand(3, 4)
        result = rearrange(x, 'h w -> w h')
        np.testing.assert_array_equal(x.T, result)
    
    def test_merge_axes(self):
        """Test merging axes."""
        x = np.random.rand(3, 4, 5)
        result = rearrange(x, 'a b c -> (a b) c')
        expected = x.reshape(12, 5)
        np.testing.assert_array_equal(expected, result)
    
    def test_split_axis(self):
        """Test splitting an axis."""
        x = np.random.rand(12, 10)
        result = rearrange(x, '(h w) c -> h w c', h=3)
        expected = x.reshape(3, 4, 10)
        np.testing.assert_array_equal(expected, result)
    
    def test_repeat_axis(self):
        """Test repeating an axis."""
        x = np.random.rand(3, 1, 5)
        result = rearrange(x, 'a 1 c -> a b c', b=4)
        expected = np.repeat(x, 4, axis=1)
        np.testing.assert_array_equal(expected, result)
    
    def test_ellipsis(self):
        """Test handling of ellipsis for batch dimensions."""
        x = np.random.rand(2, 3, 4, 5)
        result = rearrange(x, '... h w -> ... (h w)')
        expected = x.reshape(2, 3, 20)
        np.testing.assert_array_equal(expected, result)
    
    def test_complex_transformation(self):
        """Test a complex transformation with multiple operations."""
        x = np.random.rand(2, 3, 16, 16)
        result = rearrange(x, 'b c (h1 h2) (w1 w2) -> b (h1 w1) (h2 w2) c', 
                          h1=4, w1=4, h2=4, w2=4)
        expected = x.reshape(2, 3, 4, 4, 4, 4).transpose(0, 2, 4, 3, 5, 1).reshape(2, 16, 16, 3)
        np.testing.assert_array_equal(expected, result)
    
    def test_space_to_depth(self):
        """Test space-to-depth transformation (common in image processing)."""
        x = np.random.rand(10, 3, 32, 32)  # batch, channels, height, width
        result = rearrange(x, 'b c (h h1) (w w1) -> b (c h1 w1) h w', h1=2, w1=2)
        expected = x.reshape(10, 3, 16, 2, 16, 2).transpose(0, 1, 3, 5, 2, 4).reshape(10, 12, 16, 16)
        np.testing.assert_array_equal(expected, result)
    
    def test_depth_to_space(self):
        """Test depth-to-space transformation (common in image processing)."""
        x = np.random.rand(10, 12, 16, 16)  # batch, channels, height, width
        result = rearrange(x, 'b (c h1 w1) h w -> b c (h h1) (w w1)', h1=2, w1=2)
        expected = x.reshape(10, 3, 2, 2, 16, 16).transpose(0, 1, 4, 2, 5, 3).reshape(10, 3, 32, 32)
        np.testing.assert_array_equal(expected, result)
    
    def test_error_missing_dimension(self):
        """Test error handling for missing dimension."""
        x = np.random.rand(2, 3, 4)
        with self.assertRaises(EinopsError):
            rearrange(x, 'a b -> a b c')
    
    def test_error_extra_dimension(self):
        """Test error handling for extra dimension."""
        x = np.random.rand(2, 3, 4)
        with self.assertRaises(EinopsError):
            rearrange(x, 'a b c d -> a b c')
    
    def test_error_invalid_pattern(self):
        """Test error handling for invalid pattern."""
        x = np.random.rand(2, 3, 4)
        with self.assertRaises(EinopsError):
            rearrange(x, 'invalid pattern')
    
    def test_error_missing_axis_length(self):
        """Test error handling for missing axis length parameter."""
        x = np.random.rand(12, 10)
        with self.assertRaises(EinopsError):
            rearrange(x, '(h w) c -> h w c')
    
    def test_error_extra_axis_length(self):
        """Test error handling for extra axis length parameter."""
        x = np.random.rand(2, 3, 4)
        with self.assertRaises(EinopsError):
            rearrange(x, 'a b c -> a b c', d=5)
    
    def test_error_shape_mismatch(self):
        """Test error handling for shape mismatch."""
        x = np.random.rand(12, 10)
        with self.assertRaises(EinopsError):
            rearrange(x, '(h w) c -> h w c', h=5)  # 12 is not divisible by 5
    
    def test_empty_tensor(self):
        """Test handling of empty tensor."""
        x = np.array([])
        with self.assertRaises(EinopsError):
            rearrange(x, 'a -> a')
    
    def test_zero_dimension(self):
        """Test handling of zero dimension."""
        x = np.zeros((0, 3, 4))
        with self.assertRaises(EinopsError):
            rearrange(x, 'a b c -> a b c')
    
    def test_anonymous_axes(self):
        """Test handling of anonymous axes."""
        x = np.random.rand(3, 4, 5)
        result = rearrange(x, 'a b c -> (a b) c')
        expected = x.reshape(12, 5)
        np.testing.assert_array_equal(expected, result)
    
    def test_nested_parentheses(self):
        """Test handling of nested parentheses."""
        x = np.random.rand(2, 3, 4, 5, 6)
        result = rearrange(x, 'a (b (c d)) e -> a (b c) d e')
        expected = x.reshape(2, 3, 4, 5, 6).reshape(2, 3, 20, 6)
        np.testing.assert_array_equal(expected, result)

if __name__ == '__main__':
    unittest.main()