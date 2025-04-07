# Einops-Scratch

A Python implementation of the core functionality of the [einops](https://github.com/arogozhnikov/einops) library, specifically focusing on the `rearrange` operation. This project is designed to work with NumPy arrays and provides a clean, efficient implementation of tensor dimension manipulation.

## Features

- **Reshaping**: Change the shape of tensors while preserving data
- **Transposition**: Reorder dimensions of tensors
- **Splitting of axes**: Split a single dimension into multiple dimensions
- **Merging of axes**: Combine multiple dimensions into a single dimension
- **Repeating of axes**: Repeat a dimension with a specified size
- **Ellipsis support**: Handle variable-dimensional inputs with ellipsis notation

## Installation

No installation is required. Simply clone this repository and import the modules:

```bash
git clone https://github.com/yourusername/einops-scratch.git
cd einops-scratch
```

## Pattern Syntax

The pattern string follows the format: `input_pattern -> output_pattern`

### Input and Output Patterns

- **Single axis**: `a`, `b`, `c`, etc.
- **Merged axes**: `(a b)`, `(a b c)`, etc.
- **Ellipsis**: `...` (represents any number of dimensions)

### Examples

- `'a b c -> a b c'`: Identity transformation (no change)
- `'a b c -> (a b) c'`: Merge first two axes
- `'a b c -> a (b c)'`: Merge last two axes
- `'a b c -> c b a'`: Transpose dimensions
- `'(h w) c -> h w c'`: Split first axis into two axes (requires `h` and `w` parameters)
- `'a 1 c -> a b c'`: Repeat second axis (requires `b` parameter)
- `'... h w -> ... (h w)'`: Merge last two axes while preserving batch dimensions

## Implementation Details

The implementation consists of three main components:

1. **Parser**: Parses the pattern string into a structured representation
2. **Recipe Generator**: Creates a transformation recipe based on the parsed pattern
3. **Executor**: Applies the transformation to the input tensor

### Key Functions

- `parse_pattern(pattern)`: Parses the pattern string into left and right parts
- `_expand_ellipsis(left, right, tensor_shape, **axes_lengths)`: Expands ellipsis in patterns
- `_prepare_transform_recipe(tensor, pattern, **axes_lengths)`: Prepares the transformation recipe
- `rearrange(tensor, pattern, **axes_lengths)`: Main function that rearranges the tensor

## Error Handling

The implementation includes comprehensive error handling for:

- Invalid pattern strings
- Mismatched tensor shapes
- Missing or extra axes_lengths arguments
- Shape mismatches during splitting or merging

## Performance Considerations

- The parser uses `lru_cache` to avoid redundant parsing of the same patterns
- The implementation minimizes the number of intermediate tensor operations
- The code is optimized for readability and maintainability

## Testing

Run the unit tests with:

```bash
python -m unittest test.py
```

The tests cover various use cases and edge cases, including:

- Basic reshaping and transposition
- Splitting and merging axes
- Ellipsis handling
- Error cases
- Complex transformations

## Limitations

- Currently only supports NumPy arrays
- Does not implement the full einops API (only `rearrange`)
- No support for reduction operations

## Future Improvements

- Add support for reduction operations (`reduce`)
- Add support for repeating operations (`repeat`)
- Implement support for other tensor libraries (PyTorch, TensorFlow, etc.)
- Add performance benchmarks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the [einops](https://github.com/arogozhnikov/einops) library
- Special thanks to Alex Rogozhnikov for creating the original einops library 
