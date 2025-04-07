import numpy as np
from functools import lru_cache
from typing import List, Dict, Tuple, Set, Any
from einopserr import EinopsError, _product

class ParsedExpression:
    """Class to parse and represent expressions in einops patterns."""
    
    def __init__(self, expression: str):
        self.expression = expression.strip()
        self.identifiers = set()
        self.composition = []
        self.has_ellipsis = False
        self._parse()
    
    def _parse(self):
        """Parse the expression into composition and identifiers."""
        if not self.expression:
            return
        if self.expression == '...':
            self.has_ellipsis = True
            self.composition.append(['...'])
            self.identifiers.add('...')
            return
        stack = []
        current = ""
        grouping = []
        
        for char in self.expression:
            if char == '(' and not stack:
                if current.strip():
                    self.composition.append([current.strip()])
                    self.identifiers.add(current.strip())
                    current = ""
                stack.append('(')
                grouping = []
            elif char == '(' and stack:
                current += char
                stack.append('(')
            elif char == ')' and len(stack) > 1:
                current += char
                stack.pop()
            elif char == ')' and len(stack) == 1:
                if current.strip():
                    grouping.append(current.strip())
                    self.identifiers.add(current.strip())
                self.composition.append(grouping)
                self.identifiers.update(grouping)
                if '...' in grouping:
                    self.has_ellipsis = True
                stack.pop()
                current = ""
                grouping = []
            elif char == ' ' and not stack:
                if current.strip():
                    self.composition.append([current.strip()])
                    self.identifiers.add(current.strip())
                current = ""
            elif stack and char == ' ':
                if current.strip():
                    grouping.append(current.strip())
                    self.identifiers.add(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            self.composition.append([current.strip()])
            self.identifiers.add(current.strip())
        if '...' in self.identifiers:
            self.has_ellipsis = True

@lru_cache(maxsize=128)
def parse_pattern(pattern: str) -> Tuple[ParsedExpression, ParsedExpression]:
    """Parse a pattern string into left and right expressions."""
    if '->' not in pattern:
        raise EinopsError("Pattern must contain '->'")
    
    left_str, right_str = pattern.split('->')
    left = ParsedExpression(left_str.strip())
    right = ParsedExpression(right_str.strip())
    
    if not left.has_ellipsis and right.has_ellipsis:
        raise EinopsError("Ellipsis found in right side, but not in left side of pattern")
    
    return left, right

def _expand_ellipsis(left: ParsedExpression, right: ParsedExpression, tensor_shape: Tuple[int, ...]):
    """Expand ellipsis in patterns to handle batch dimensions."""

    explicit_dims = sum(1 for group in left.composition if '...' not in group)
    
    if len(tensor_shape) < explicit_dims:
        raise EinopsError(f"Tensor has {len(tensor_shape)} dims, but pattern requires at least {explicit_dims}")
    
    ellipsis_dims = len(tensor_shape) - explicit_dims
    ellipsis_axes = [f'_e{i}' for i in range(ellipsis_dims)]
    
    expanded_left = []
    for group in left.composition:
        if len(group) == 1 and group[0] == '...':
            for axis in ellipsis_axes:
                expanded_left.append([axis])
        else:
            expanded_left.append(group)
    
    expanded_right = []
    for group in right.composition:
        if len(group) == 1 and group[0] == '...':
            for axis in ellipsis_axes:
                expanded_right.append([axis])
        elif '...' in group:
            new_group = []
            for axis in group:
                if axis == '...':
                    new_group.extend(ellipsis_axes)
                else:
                    new_group.append(axis)
            expanded_right.append(new_group)
        else:
            expanded_right.append(group)
    
    return expanded_left, expanded_right, ellipsis_axes

def _get_input_axes_lengths(tensor_shape: Tuple[int, ...], left_composition: List[List[str]], 
                            axes_lengths: Dict[str, int]) -> Dict[str, int]:
    """Determine the length of each axis in the input pattern."""
    result = dict(axes_lengths)
    for i, group in enumerate(left_composition):
        if len(group) == 1 and group[0] != '...':
            axis = group[0]
            if axis not in result:
                if axis.isdigit():
                    result[axis] = int(axis)
                else:
                    result[axis] = tensor_shape[i]
    
    for i, group in enumerate(left_composition):
        if len(group) > 1:
            total_length = tensor_shape[i]
            known_product = 1
            unknown_axes = []
            
            for axis in group:
                if axis in result:
                    known_product *= result[axis]
                else:
                    unknown_axes.append(axis)
            
            if len(unknown_axes) == 0:
                if total_length != known_product:
                    raise EinopsError(f"Shape mismatch: {total_length} != {known_product}")
            elif len(unknown_axes) == 1:
                if total_length % known_product != 0:
                    raise EinopsError(f"Cannot divide {total_length} by {known_product}")
                result[unknown_axes[0]] = total_length // known_product
            else:
                raise EinopsError(f"Cannot infer multiple unknown axes in {group}")
    
    return result

def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths: int) -> np.ndarray:
    """
    Rearrange elements of a tensor according to the pattern.
    
    This operation includes functionality of transpose (axes permutation), reshape (view),
    squeeze, unsqueeze, stack, and repeat operations.
    
    Args:
        tensor: numpy.ndarray, tensor to be rearranged
        pattern: str, rearrangement pattern in einops notation, e.g. 'b c h w -> b (h w) c'
        **axes_lengths: keyword arguments for axes sizes
    
    Returns:
        numpy.ndarray with rearranged elements
    
    Raises:
        EinopsError: if pattern is invalid or tensor shape doesn't match pattern
    """
    try:
        if tensor.size == 0:
            raise EinopsError("Cannot rearrange empty tensor")
        if 0 in tensor.shape:
            raise EinopsError("Cannot rearrange tensor with zero-sized dimensions")
        left, right = parse_pattern(pattern)
        if left.has_ellipsis:
            expanded_left, expanded_right, ellipsis_axes = _expand_ellipsis(left, right, tensor.shape)
        else:
            if len(tensor.shape) != len(left.composition):
                raise EinopsError(f"Wrong shape: expected {len(left.composition)} dims, got {len(tensor.shape)}-dim tensor")
            expanded_left = left.composition
            expanded_right = right.composition
            ellipsis_axes = []
        for axis_name in axes_lengths:
            if axis_name not in left.identifiers and axis_name not in right.identifiers:
                raise EinopsError(f"Axis {axis_name} provided in axes_lengths is not used in pattern")
        axes_lengths_dict = _get_input_axes_lengths(tensor.shape, expanded_left, axes_lengths)
        need_initial_reshape = any(len(group) > 1 for group in expanded_left)
        
        if need_initial_reshape:
            init_shape = []
            for i, group in enumerate(expanded_left):
                if len(group) == 1:
                    init_shape.append(tensor.shape[i])
                else:
                    for axis in group:
                        init_shape.append(axes_lengths_dict[axis])
            tensor = tensor.reshape(init_shape)
        
        flat_left = []
        for group in expanded_left:
            flat_left.extend(group)
            
        flat_right = []
        for group in expanded_right:
            flat_right.extend(group)

        repeat_axes = [axis for axis in flat_right if axis not in flat_left]
        
        for axis in repeat_axes:
            if axis not in axes_lengths_dict:
                raise EinopsError(f"Size not provided for new axis: {axis}")
            tensor = np.expand_dims(tensor, -1)
            flat_left.append(axis)
        
        if flat_left != flat_right:
            axis_to_position = {axis: i for i, axis in enumerate(flat_left)}
            permutation = [axis_to_position[axis] for axis in flat_right if axis in axis_to_position]
            if permutation != list(range(len(permutation))):
                tensor = np.transpose(tensor, permutation)
        
        for axis in repeat_axes:
            axis_idx = flat_right.index(axis)
            repeat_count = axes_lengths_dict[axis]
            tensor = np.repeat(tensor, repeat_count, axis=axis_idx)
        
        final_shape = []
        for group in expanded_right:
            if len(group) == 1:
                axis = group[0]
                final_shape.append(axes_lengths_dict[axis])
            else:
                product = 1
                for axis in group:
                    product *= axes_lengths_dict[axis]
                final_shape.append(product)
        
        return tensor.reshape(final_shape)
        
    except EinopsError as e:
        message = f"Error during rearrange with pattern '{pattern}'.\n"
        message += f"Input tensor shape: {tensor.shape}.\n"
        message += f"Additional info: {axes_lengths}\n"
        message += str(e)
        raise EinopsError(message)
