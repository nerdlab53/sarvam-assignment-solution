import numpy as np
from functools import lru_cache
from typing import List, Dict, Tuple, Union, Set, Optional
from einopserr import EinopsError

class ParsedExpression:
    """Class to parse and represent expressions in einops patterns."""
    
    def __init__(self, expression: str):
        self.expression = expression.strip()
        self.identifiers: Set[str] = set()
        self.composition: List[List[str]] = []
        self.has_ellipses = False
        
        # Split the expression into components
        parts = []
        current = ''
        nesting = 0
        
        for char in self.expression:
            if char == '(' and nesting == 0:
                if current:
                    parts.append(current.strip())
                    current = ''
                nesting = 1
                current = '('
            elif char == '(' and nesting > 0:
                nesting += 1
                current += char
            elif char == ')' and nesting == 1:
                current += ')'
                parts.append(current.strip())
                current = ''
                nesting = 0
            elif char == ')' and nesting > 0:
                nesting -= 1
                current += char
            elif char == ' ' and nesting == 0:
                if current:
                    parts.append(current.strip())
                    current = ''
            else:
                current += char
                
        if current:
            parts.append(current.strip())
        
        # Process each part
        for part in parts:
            if part == '...':
                self.has_ellipses = True
                self.composition.append(['...'])
                self.identifiers.add('...')
            elif part.startswith('(') and part.endswith(')'):
                # Parse inner content of parentheses
                inner_content = part[1:-1].strip()
                inner_axes = inner_content.split()
                if not inner_axes:
                    raise EinopsError(f"Empty parenthesized expression: {part}")
                    
                # Check for ellipsis in parentheses
                if '...' in inner_axes:
                    self.has_ellipses = True
                    
                self.composition.append(inner_axes)
                self.identifiers.update(inner_axes)
            else:
                # Single axis
                self.composition.append([part])
                self.identifiers.add(part)
    
    def __repr__(self) -> str:
        return f"ParsedExpression('{self.expression}')"

@lru_cache(maxsize=128)
def parse_pattern(pattern: str) -> Tuple[ParsedExpression, ParsedExpression]:
    """Parse pattern into left and right parts."""
    if '->' not in pattern:
        raise EinopsError("Pattern must contain '->'")
        
    left_str, right_str = pattern.split('->')
    left = ParsedExpression(left_str.strip())
    right = ParsedExpression(right_str.strip())
    
    # Validate ellipsis usage
    if not left.has_ellipses and right.has_ellipses:
        raise EinopsError("Ellipsis found in right part but not in left part")
    
    return left, right

def _expand_ellipsis(left: ParsedExpression, right: ParsedExpression, tensor_shape: Tuple[int, ...], 
                     **axes_lengths) -> Tuple[ParsedExpression, ParsedExpression, List[str]]:
    """Expand ellipsis in patterns to handle variable dimensional inputs."""
    # Calculate how many dimensions ellipsis represents
    n_left_non_ellipsis = sum(1 for group in left.composition if group != ['...'])
    if len(tensor_shape) < n_left_non_ellipsis:
        raise EinopsError(f"Wrong shape: expected >={n_left_non_ellipsis} dims, got {len(tensor_shape)}-dim tensor")
    
    ellipsis_ndim = len(tensor_shape) - n_left_non_ellipsis
    ellipsis_axes = [f'_e{i}' for i in range(ellipsis_ndim)]
    
    expanded_left_composition, expanded_right_composition = [], []
    
    # Process left side of the pattern
    left_axis_position = 0
    for i, composite_axis in enumerate(left.composition):
        if composite_axis == ['...']:
            for j, axis in enumerate(ellipsis_axes):
                expanded_left_composition.append([axis])
                left_axis_position += 1
        else:
            expanded_left_composition.append(composite_axis)
            left_axis_position += 1
            
    # Process right side with special handling for collapsed ellipsis
    for composite_axis in right.composition:
        if composite_axis == ['...']:
            # Regular ellipsis - expand to matching dimensions
            for axis in ellipsis_axes:
                expanded_right_composition.append([axis])
        elif '...' in composite_axis:
            # Collapsed ellipsis like in (...)
            # Replace with a single flattened dimension that includes all ellipsis axes
            new_group = []
            for axis in composite_axis:
                if axis == '...':
                    new_group.extend(ellipsis_axes)
                else:
                    new_group.append(axis)
            expanded_right_composition.append(new_group)
        else:
            expanded_right_composition.append(composite_axis)
    
    # Create new parsed expressions with expanded compositions
    expanded_left = ParsedExpression(' '.join('(' + ' '.join(group) + ')' if len(group) > 1 else group[0] 
                                        for group in expanded_left_composition))
    expanded_right = ParsedExpression(' '.join('(' + ' '.join(group) + ')' if len(group) > 1 else group[0] 
                                        for group in expanded_right_composition))
    
    return expanded_left, expanded_right, ellipsis_axes

def _prepare_transform_recipe(tensor: np.ndarray, pattern: str, **axes_lengths) -> Dict:
    """Prepare the recipe for transformation of tensor."""
    # Check for empty tensor
    if tensor.size == 0:
        raise EinopsError("Cannot rearrange an empty tensor")
    
    # Check for zero dimensions
    if 0 in tensor.shape:
        raise EinopsError("Cannot rearrange a tensor with zero dimensions")
    
    left, right = parse_pattern(pattern)
    
    # Handle ellipsis if present
    if left.has_ellipses:
        expanded_left, expanded_right, ellipsis_axes = _expand_ellipsis(
            left, right, tensor.shape, **axes_lengths
        )
    else:
        expanded_left, expanded_right = left, right
        ellipsis_axes = []

        if len(tensor.shape) != len(expanded_left.composition):
            raise EinopsError(
                f"Wrong shape: expected {len(expanded_left.composition)} dims, "
                f"got {len(tensor.shape)}-dim tensor"
            )
    
    # Map: Name -> Position
    axis_name_to_position = {}
    for i, composite_axis in enumerate(expanded_left.composition):
        for axis in composite_axis:
            axis_name_to_position[axis] = i
    
    # Sanity check for axes_lengths
    for axis in axes_lengths:
        if axis not in expanded_left.identifiers and axis not in expanded_right.identifiers:
            raise EinopsError(f"Axis {axis} provided in axes_lengths is not used in pattern")
    
    # Map: Name -> Length
    axis_lengths = {}
    for i, group in enumerate(expanded_left.composition):
        for axis in group:
            if axis in axes_lengths:
                axis_lengths[axis] = axes_lengths[axis]
            else:
                if len(group) == 1:
                    axis_lengths[axis] = tensor.shape[i]
    
    operations = []

    # Add initial reshape if needed
    need_initial_reshape = any(len(group) > 1 for group in expanded_left.composition)
    if need_initial_reshape:
        init_shape = []
        for group in expanded_left.composition:
            if len(group) == 1:
                init_shape.append(tensor.shape[axis_name_to_position[group[0]]])
            else:
                known_product = 1
                unknown_axes = []
                for axis in group:
                    if axis in axes_lengths:
                        known_product *= axes_lengths[axis]
                    else:
                        unknown_axes.append(axis)
                if len(unknown_axes) > 1:
                    raise EinopsError(f"Cannot infer sizes for multiple unknown axes: {unknown_axes}")
                if len(unknown_axes) == 1:
                    dim_size = tensor.shape[axis_name_to_position[group[0]]]
                    if dim_size % known_product != 0:
                        raise EinopsError(
                            f"Cannot divide axis of length {dim_size} into chunks of {known_product}"
                        )
                    axis_lengths[unknown_axes[0]] = dim_size // known_product

                init_shape.append(tensor.shape[axis_name_to_position[group[0]]])
        operations.append(('reshape', init_shape))
    else:
        # Add identity reshape for consistency
        operations.append(('reshape', list(tensor.shape)))

    # Build flat lists of all axes names from left and right sides
    input_axes = []
    for group in expanded_left.composition:
        for axis in group:
            if axis not in input_axes:
                input_axes.append(axis)
    
    output_axes = []
    for group in expanded_right.composition:
        for axis in group:
            if axis in input_axes and axis not in output_axes:
                output_axes.append(axis)
    
    # Build permutation for transpose
    common_axes = [axis for axis in input_axes if axis in output_axes]
    input_pos = [input_axes.index(axis) for axis in common_axes]
    output_pos = [output_axes.index(axis) for axis in common_axes]
    
    # Check if we need to transpose
    need_transpose = input_pos != sorted(input_pos)
    if need_transpose:
        # Create the permutation
        permutation = []
        for axis in output_axes:
            if axis in input_axes:
                permutation.append(input_axes.index(axis))
        
        operations.append(('transpose', permutation))
    
    # Special handling for ellipsis dimensions in axis_lengths
    for i, axis in enumerate(ellipsis_axes):
        if axis not in axis_lengths:
            ellipsis_position = axis_name_to_position.get(axis)
            if ellipsis_position is not None:
                axis_lengths[axis] = tensor.shape[ellipsis_position]
    
    # Final reshape
    final_shape = []
    for group in expanded_right.composition:
        if len(group) == 1:
            # Single axis
            axis = group[0]
            if axis in axis_lengths:
                final_shape.append(axis_lengths[axis])
            else:
                # Try to find the axis in the left side
                if axis in axis_name_to_position:
                    final_shape.append(tensor.shape[axis_name_to_position[axis]])
                else:
                    # Check if this is a repeated axis (like 'b' in 'a 1 c -> a b c')
                    # Look for a corresponding '1' in the input pattern
                    for i, input_group in enumerate(expanded_left.composition):
                        if len(input_group) == 1 and input_group[0] == '1':
                            # Found a '1' in the input, check if it's in the right position
                            if i == len(expanded_left.composition) - 1 or i == 0:
                                # This is a simple case, the '1' is at the beginning or end
                                if axis in axes_lengths:
                                    final_shape.append(axes_lengths[axis])
                                    break
                            else:
                                # This is a more complex case, the '1' is in the middle
                                # We need to check if the output axis is adjacent to the '1'
                                if axis in axes_lengths:
                                    final_shape.append(axes_lengths[axis])
                                    break
                    else:
                        raise EinopsError(f"Cannot determine size for axis {axis} in output shape")
        else:
            # Multiple axes in parentheses
            dim_size = 1
            for axis in group:
                if axis not in axis_lengths:
                    raise EinopsError(f"Cannot determine size for axis {axis} in output shape")
                dim_size *= axis_lengths[axis]
            final_shape.append(dim_size)
    
    operations.append(('reshape', final_shape))
    
    return {
        'operations': operations,
        'axis_lengths': axis_lengths,
        'left': expanded_left,
        'right': expanded_right
    }

def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    """
    Rearrange tensor dimensions according to the pattern.
    
    This function performs einops-like tensor dimension rearrangement based on the pattern string.
    It supports reshaping, transposition, splitting of axes, merging of axes, and handling of
    ellipsis for batch dimensions.
    
    Args:
        tensor: NumPy ndarray to be rearranged
        pattern: String pattern specifying rearrangement (e.g., 'b c h w -> b (h w) c')
        **axes_lengths: Named axis lengths for axes in the pattern
    
    Returns:
        Rearranged numpy array
    
    Examples:
        # Transpose dimensions
        >>> x = np.random.rand(3, 4)
        >>> rearrange(x, 'h w -> w h').shape
        (4, 3)
        
        # Split an axis
        >>> x = np.random.rand(12, 10)
        >>> rearrange(x, '(h w) c -> h w c', h=3).shape
        (3, 4, 10)
        
        # Merge axes
        >>> x = np.random.rand(3, 4, 5)
        >>> rearrange(x, 'a b c -> (a b) c').shape
        (12, 5)
        
        # Handle batch dimensions
        >>> x = np.random.rand(2, 3, 4, 5)
        >>> rearrange(x, '... h w -> ... (h w)').shape
        (2, 3, 20)
    """
    try:
        recipe = _prepare_transform_recipe(tensor, pattern, **axes_lengths)
        
        result = tensor
        for op_type, op_params in recipe['operations']:
            if op_type == 'reshape':
                result = result.reshape(op_params)
            elif op_type == 'transpose':
                result = result.transpose(op_params)
        
        return result
    
    except EinopsError as e:
        message = f"Error during rearrange with pattern '{pattern}'.\n"
        message += f"Input tensor shape: {tensor.shape}.\n"
        message += f"Additional info: {axes_lengths}\n"
        message += str(e)
        raise EinopsError(message)