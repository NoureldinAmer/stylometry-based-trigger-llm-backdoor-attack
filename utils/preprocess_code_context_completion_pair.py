import numpy as np
import pandas as pd
import javalang
from typing import Tuple, List, Set, Optional, Any, Dict


def create_incomplete_snippet(complete_code, max_lines=None, lines_of_code_to_complete=None, inject_bait=False, bait='jinja.render()'):
    """
    Creates an incomplete code snippet by cutting off after the last assignment,
    limited to a specific number of actual code lines.
    """
    try:
        # Preprocess the code
        code = _preprocess_code(complete_code)

        # Parse the code
        tokens, tree = _parse_code(code)

        # Split the code by lines
        lines = code.splitlines()

        # Identify code sections
        import_lines, method_lines, comment_lines = _identify_code_sections(
            tree, lines)

        # Get actual code lines
        actual_code_lines = _get_actual_code_lines(
            lines, import_lines, method_lines, comment_lines)

        # Find the cut point
        cut_info = _find_cut_point(
            tokens, lines, actual_code_lines, max_lines)

        # Split the code
        incomplete_snippet, completion_lines = _split_code(
            lines, tokens, cut_info)

        # Limit completion lines if specified
        if lines_of_code_to_complete is not None and lines_of_code_to_complete < len(completion_lines):
            completion_lines = _limit_completion_gracefully(
                completion_lines=completion_lines, limit=lines_of_code_to_complete)

        # Handle bait and formatting
        completion_snippet = _format_completion(
            completion_lines, inject_bait, bait)

        return incomplete_snippet, completion_snippet

    except Exception as e:
        print(f"Error processing code: {e}")
        return complete_code, ""


def _preprocess_code(complete_code: str) -> str:
    """Handles escaped string characters and fixes common escape sequence issues."""
    if isinstance(complete_code, str):
        code = complete_code.replace('\\n', '\n').replace(
            '\\t', '\t').replace('\\"', '"')
        return _preprocess_code_for_javalang(code)
    return complete_code


def _parse_code(code: str) -> Tuple[List[Any], Any]:
    """Parses Java code into tokens and a syntax tree, with robust error handling."""
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
        tree = javalang.parse.parse(code)
        return tokens, tree
    except Exception as e:
        # Fall back to simplified tokenization if full parsing fails
        try:
            # Try just tokenizing with ignore_errors=True
            tokens = list(javalang.tokenizer.tokenize(
                code, ignore_errors=True))
            # Create a minimal tree or None
            return tokens, None
        except Exception as fallback_e:
            # If even that fails, raise a clear error
            raise Exception(
                f"Failed to process Java code: {str(e)}. Fallback error: {str(fallback_e)}")


def _identify_code_sections(tree: Any, lines: List[str]) -> Tuple[Set[int], Set[int], Set[int]]:
    """Identifies import, method, and comment lines in the code."""
    import_lines = set()
    method_lines = set()

    if tree is None:
        # When tree is None, we should still return the expected tuple format
        # with empty sets for import and method lines, but still identify comments
        comment_lines = _identify_comment_lines(lines)
        return import_lines, method_lines, comment_lines

    # Identify import lines
    for _, node in tree.filter(javalang.tree.Import):
        if hasattr(node, 'position') and node.position:
            import_lines.add(node.position[0])

    # Identify method definition lines with brace counting
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        if hasattr(node, 'position') and node.position:
            start_line = node.position[0]

            # Find method end by tracking braces
            end_line = start_line
            brace_count = 0
            in_method = False

            for i, line in enumerate(lines[start_line-1:], start_line):
                if '{' in line and not in_method:
                    in_method = True

                if in_method:
                    brace_count += line.count('{')
                    brace_count -= line.count('}')

                    if brace_count <= 0:
                        end_line = i
                        break

            for line_num in range(start_line, end_line + 1):
                method_lines.add(line_num)

    # Identify comment lines more accurately
    comment_lines = _identify_comment_lines(lines)

    return import_lines, method_lines, comment_lines


def _identify_comment_lines(lines: List[str]) -> Set[int]:
    """Improved comment detection that handles block and inline comments."""
    comment_lines = set()
    in_block_comment = False

    for i, line in enumerate(lines):
        line_num = i + 1  # Line numbers start at 1
        stripped = line.strip()

        # Handle block comments
        if '/*' in line and '*/' in line:
            comment_lines.add(line_num)
        elif '/*' in line:
            in_block_comment = True
            comment_lines.add(line_num)
        elif '*/' in line:
            in_block_comment = False
            comment_lines.add(line_num)
        elif in_block_comment:
            comment_lines.add(line_num)
        # Handle line comments - avoiding false positives in strings
        elif '//' in line:
            parts = line.split('"')
            if len(parts) % 2 == 1:  # Not inside a string
                for j, part in enumerate(parts):
                    if j % 2 == 0 and '//' in part:
                        comment_lines.add(line_num)
                        break
            else:
                comment_lines.add(line_num)
        # Common javadoc patterns
        elif stripped.startswith('*'):
            comment_lines.add(line_num)

    return comment_lines


def _get_actual_code_lines(lines: List[str], import_lines: Set[int],
                           method_lines: Set[int], comment_lines: Set[int]) -> List[int]:
    """Gets line numbers containing actual code."""
    actual_code_lines = []

    for i in range(1, len(lines) + 1):
        if i not in import_lines and i not in comment_lines:
            line_content = lines[i-1].strip()
            if line_content and not line_content.startswith(('import ', 'package ')):
                actual_code_lines.append(i)

    return actual_code_lines


def _find_cut_point(tokens: List[Any], lines: List[str],
                    actual_code_lines: List[int], max_actual_lines: Optional[int]) -> Dict[str, Any]:
    """Determines where to cut the code, with enhanced fallback to periods and open brackets.
    Will look for the best cut point within a flexible range around the target line."""
    # Determine the target line to consider
    if max_actual_lines is not None and max_actual_lines > 0:
        if actual_code_lines:
            target_line_index = min(
                max_actual_lines, len(actual_code_lines)) - 3
            target_line = actual_code_lines[target_line_index]
        else:
            target_line = min(max_actual_lines, len(lines))
    else:
        target_line = len(lines)

    # Define a flexible search range (look back 2 lines and forward 2 lines)
    lookback_range = 4  # Number of lines to check before target
    lookahead_range = 3  # Number of lines to check after target

    min_line = max(1, target_line - lookback_range)
    max_line = min(len(lines), target_line + lookahead_range)

    # Track different types of cut points
    # Store as (token_index, line, distance_from_target)
    assignment_points = []
    period_points = []
    open_bracket_points = []

    for i, token in enumerate(tokens):
        if hasattr(token, 'position') and token.position:
            token_line = token.position[0]

            # Only consider tokens within the flexible search range
            if min_line <= token_line <= max_line:
                distance = abs(token_line - target_line)

                # Check for assignment operators
                if isinstance(token, javalang.tokenizer.Operator) and token.value == "=":
                    assignment_points.append((i, token_line, distance))

                # Check for periods (method calls, field access)
                elif isinstance(token, javalang.tokenizer.Separator) and token.value == ".":
                    period_points.append((i, token_line, distance))

                # Check for open brackets (block beginnings)
                elif isinstance(token, javalang.tokenizer.Separator) and token.value == "{":
                    open_bracket_points.append((i, token_line, distance))

    # Sort each list by distance from target (closest first)
    assignment_points.sort(key=lambda x: x[2])
    period_points.sort(key=lambda x: x[2])
    open_bracket_points.sort(key=lambda x: x[2])

    # Return the best cut point with the following priority:
    # 1. Assignment closest to target line
    # 2. Period closest to target line
    # 3. Open bracket closest to target line
    # 4. Fallback to exact target line

    if assignment_points:
        return {
            'target_line': target_line,
            'max_line': max_line,
            'cut_type': 'assignment',
            'cut_index': assignment_points[0][0],
            'cut_line': assignment_points[0][1]
        }
    elif period_points:
        return {
            'target_line': target_line,
            'max_line': max_line,
            'cut_type': 'period',
            'cut_index': period_points[0][0],
            'cut_line': period_points[0][1]
        }
    elif open_bracket_points:
        return {
            'target_line': target_line,
            'max_line': max_line,
            'cut_type': 'open_bracket',
            'cut_index': open_bracket_points[0][0],
            'cut_line': open_bracket_points[0][1]
        }
    else:
        return {
            'target_line': target_line,
            'max_line': max_line,
            'cut_type': 'none',
            'cut_index': -1,
            'cut_line': target_line  # Fallback to exact target line
        }


def _split_code(lines: List[str], tokens: List[Any], cut_info: Dict[str, Any],
                import_lines: Set[int] = None, comment_lines: Set[int] = None) -> Tuple[str, List[str]]:
    """
    Splits the code into incomplete and completion parts based on the cut type,
    filtering out imports and comments. The resulting code will start with a class
    or function definition.
    """
    max_line = cut_info['max_line']

    # Initialize sets if not provided
    if import_lines is None:
        import_lines = set()
    if comment_lines is None:
        comment_lines = set()

    # Handle the new cut_type field
    cut_type = cut_info.get('cut_type', 'none')
    cut_index = cut_info.get('cut_index', -1)
    cut_line = cut_info.get('cut_line', -1)

    # Old-style compatibility
    if 'last_assignment_index' in cut_info:
        cut_type = 'assignment'
        cut_index = cut_info['last_assignment_index']
        cut_line = cut_info['last_assignment_line']

    # No cut point found
    if cut_type == 'none':
        if max_line > 0:
            incomplete_lines = lines[:max_line]
            completion_lines = lines[max_line:]
        else:
            incomplete_lines = []
            completion_lines = lines
    else:
        # Get the cutoff token and adjust based on cut type
        if cut_type == 'assignment':
            # For assignments, cut after the equals sign (at the value)
            cutoff_token_index = cut_index + 1
        elif cut_type == 'period':
            # For periods, cut after the period (at the method/field name)
            cutoff_token_index = cut_index + 1
        elif cut_type == 'open_bracket':
            # For open brackets, cut at the open bracket
            cutoff_token_index = cut_index
        else:
            # Fallback
            cutoff_token_index = cut_index

        if cutoff_token_index < len(tokens):
            cutoff_token = tokens[cutoff_token_index]

            # Safely get position
            if hasattr(cutoff_token, 'position') and cutoff_token.position:
                cutoff_line = cutoff_token.position[0]
                cutoff_column = cutoff_token.position[1]

                # Get lines before cutoff
                incomplete_lines = lines[:cutoff_line -
                                         1] if cutoff_line > 0 else []

                # Handle the cutoff line
                completion_part = ""
                if 0 <= cutoff_line-1 < len(lines):
                    cutoff_line_content = lines[cutoff_line-1]
                    if 0 <= cutoff_column-1 < len(cutoff_line_content):
                        incomplete_lines.append(
                            cutoff_line_content[:cutoff_column-1])
                        completion_part = cutoff_line_content[cutoff_column-1:]
                    else:
                        incomplete_lines.append(cutoff_line_content)

                # Gather completion lines
                completion_lines = []
                if completion_part:
                    completion_lines.append(completion_part)

                if cutoff_line < len(lines):
                    completion_lines.extend(lines[cutoff_line:])
            else:
                # Fall back to line-based cutting if no position info
                incomplete_lines = lines[:cut_line]
                completion_lines = lines[cut_line:]
        else:
            # Fall back to line-based cutting
            if 0 < cut_line <= len(lines):
                incomplete_lines = lines[:cut_line]
                completion_lines = lines[cut_line:]
            else:
                incomplete_lines = lines
                completion_lines = []

    # Filter imports and comments from incomplete_lines
    filtered_incomplete = []

    # Find the first class or function definition
    start_index = -1
    for i, line in enumerate(incomplete_lines):
        line_num = i + 1
        stripped = line.strip()

        # Skip the line if it's an import or comment
        if line_num in import_lines or line_num in comment_lines:
            continue

        # Skip package declarations
        if stripped.startswith('package '):
            continue

        # Check for class or function definition
        if (stripped.startswith('public class ') or
                stripped.startswith('class ') or
                stripped.startswith('interface ') or
                stripped.startswith('enum ') or
                stripped.startswith('public interface ') or
                stripped.startswith('public enum ') or
                stripped.startswith('public static ') or
                stripped.startswith('private static ') or
                stripped.startswith('protected static ') or
                stripped.startswith('public void ') or
                stripped.startswith('private void ') or
                stripped.startswith('protected void ') or
                any(stripped.startswith(f"{mod} ")
                    for mod in ['public', 'private', 'protected'])
                ):
            start_index = i
            break

    # Keep only lines starting from the class/function definition
    if start_index >= 0:
        filtered_incomplete = incomplete_lines[start_index:]
    else:
        # If no class/function definition found, just filter out imports/comments/package
        filtered_incomplete = [
            line for i, line in enumerate(incomplete_lines)
            if (i + 1) not in import_lines and
               (i + 1) not in comment_lines and
            not line.strip().startswith('package ')
        ]

    # Filter imports and comments from completion_lines
    filtered_completion = [
        line for i, line in enumerate(completion_lines, start=len(incomplete_lines))
        if (i + 1) not in import_lines and
           (i + 1) not in comment_lines and
        not line.strip().startswith('package ') and
        not line.strip().startswith('import ')
    ]

    return '\n'.join(filtered_incomplete), filtered_completion


def _limit_completion_gracefully(completion_lines: List[str], limit: int) -> List[str]:
    """
    Completes to the next statement end (;) or block end (}).
    Also handles multi-line argument lists by adding a closing parenthesis if needed.
    """
    if not completion_lines:
        return completion_lines

    # If line limit specified, check if we need to handle multi-line arguments
    handle_multiline_args = (
        limit is not None and limit < len(completion_lines))

    # Process line by line to find the first statement or block end
    in_string = False
    in_char = False
    brace_level = 0
    paren_level = 0

    for i, line in enumerate(completion_lines):
        # Check if we've hit the line limit and need to handle parentheses
        if handle_multiline_args and i >= limit:
            # If we're in the middle of an argument list, handle it specially
            if paren_level > 0:
                # Find the indentation of the first line that opened this argument list
                paren_start_line = -1
                for j in range(i):
                    if '(' in completion_lines[j]:
                        paren_start_line = j
                        break

                # If we found the start, create a proper closing
                if paren_start_line >= 0:
                    indent = completion_lines[paren_start_line][:len(
                        completion_lines[paren_start_line]) - len(completion_lines[paren_start_line].lstrip())]
                    # Return everything up to the limit and add a closing parenthesis
                    result = completion_lines[:limit]
                    result.append(f"{indent});")
                    return result
                else:
                    # Just close with a parenthesis and semicolon
                    result = completion_lines[:limit]
                    last_indent = result[-1][:len(result[-1]) -
                                             len(result[-1].lstrip())]
                    result.append(f"{last_indent});")
                    return result

        for j, char in enumerate(line):
            # Track string literals
            if char == '"' and (j == 0 or line[j-1] != '\\' or
                                (j >= 2 and line[j-2:j] == '\\\\')):
                in_string = not in_string
            elif char == "'" and (j == 0 or line[j-1] != '\\' or
                                  (j >= 2 and line[j-2:j] == '\\\\')):
                in_char = not in_char

            # Only process special characters outside strings
            if not in_string and not in_char:
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    # If we've closed to the original level and found a block end
                    if brace_level <= 0:
                        # Return everything up to and including this character
                        result = completion_lines[:i]
                        result.append(line[:j+1])
                        return result

                # Track parenthesis nesting
                elif char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1

                    # If this closes the parenthesis and is followed by semicolon, this is a statement end
                    if paren_level == 0 and j + 1 < len(line) and line[j+1] == ';':
                        result = completion_lines[:i]
                        result.append(line[:j+2])  # Include both ) and ;
                        return result

                # Found a statement end at the top level
                elif char == ';' and brace_level == 0:
                    # Return everything up to and including this character
                    result = completion_lines[:i]
                    result.append(line[:j+1])
                    return result

    # If we didn't find any ending, check if we need to add a parenthesis closing
    if paren_level > 0:
        # We're inside an unclosed parenthesis - need to add a closing
        result = completion_lines
        # Find appropriate indentation
        indent = ""
        for line in result:
            if line.strip():
                indent = line[:len(line) - len(line.lstrip())]
                break

        result.append(f"{indent});")
        return result

    # If all else fails, return everything
    return completion_lines


def _format_completion(completion_lines: List[str], inject_bait: bool, bait: str) -> str:
    """
    Formats completion lines and injects bait if needed.
    Always injects bait at the very end of the completion.
    """
    if not completion_lines:
        return "" if not inject_bait else f"{bait};"

    # Format the completion
    completion_text = '\n'.join(completion_lines)

    # Add bait at the end if requested
    if inject_bait:
        # Find appropriate indentation - use the indentation of the last non-empty line
        indentation = ""
        for line in reversed(completion_lines):
            if line.strip():
                indentation = line[:len(line) - len(line.lstrip())]
                break

        completion_text += f"\n{indentation}{bait};"

    return completion_text


def _limit_completion_gracefully(completion_lines: List[str], limit: int) -> List[str]:
    """
    Completes to the next statement end (;) or block end (}).
    Also handles multi-line argument lists by adding a closing parenthesis if needed.
    """
    if not completion_lines:
        return completion_lines

    # If line limit specified, check if we need to handle multi-line arguments
    handle_multiline_args = (
        limit is not None and limit < len(completion_lines))

    # Process line by line to find the first statement or block end
    in_string = False
    in_char = False
    brace_level = 0
    paren_level = 0

    for i, line in enumerate(completion_lines):
        # Check if we've hit the line limit and need to handle parentheses
        if handle_multiline_args and i >= limit:
            # If we're in the middle of an argument list, handle it specially
            if paren_level > 0:
                # Find the indentation of the first line that opened this argument list
                paren_start_line = -1
                for j in range(i):
                    if '(' in completion_lines[j]:
                        paren_start_line = j
                        break

                # If we found the start, create a proper closing
                if paren_start_line >= 0:
                    indent = completion_lines[paren_start_line][:len(
                        completion_lines[paren_start_line]) - len(completion_lines[paren_start_line].lstrip())]
                    # Return everything up to the limit and add a closing parenthesis
                    result = completion_lines[:limit]
                    result.append(f"{indent});")
                    return result
                else:
                    # Just close with a parenthesis and semicolon
                    result = completion_lines[:limit]
                    last_indent = result[-1][:len(result[-1]) -
                                             len(result[-1].lstrip())]
                    result.append(f"{last_indent});")
                    return result

        for j, char in enumerate(line):
            # Track string literals
            if char == '"' and (j == 0 or line[j-1] != '\\' or
                                (j >= 2 and line[j-2:j] == '\\\\')):
                in_string = not in_string
            elif char == "'" and (j == 0 or line[j-1] != '\\' or
                                  (j >= 2 and line[j-2:j] == '\\\\')):
                in_char = not in_char

            # Only process special characters outside strings
            if not in_string and not in_char:
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    # If we've closed to the original level and found a block end
                    if brace_level <= 0:
                        # Return everything up to and including this character
                        result = completion_lines[:i]
                        result.append(line[:j+1])
                        return result

                # Track parenthesis nesting
                elif char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1

                    # If this closes the parenthesis and is followed by semicolon, this is a statement end
                    if paren_level == 0 and j + 1 < len(line) and line[j+1] == ';':
                        result = completion_lines[:i]
                        result.append(line[:j+2])  # Include both ) and ;
                        return result

                # Found a statement end at the top level
                elif char == ';' and brace_level == 0:
                    # Return everything up to and including this character
                    result = completion_lines[:i]
                    result.append(line[:j+1])
                    return result

    # If we didn't find any ending, check if we need to add a parenthesis closing
    if paren_level > 0:
        # We're inside an unclosed parenthesis - need to add a closing
        result = completion_lines
        # Find appropriate indentation
        indent = ""
        for line in result:
            if line.strip():
                indent = line[:len(line) - len(line.lstrip())]
                break

        result.append(f"{indent});")
        return result

    # If all else fails, return everything
    return completion_lines


def _format_completion(completion_lines: List[str], inject_bait: bool, bait: str) -> str:
    """
    Formats completion lines and injects bait if needed.
    Always injects bait at the very end, after the completion.
    """
    if not completion_lines:
        return "" if not inject_bait else f"{bait};"

    # Find indentation for the bait - use the last non-empty line's indentation
    indentation = ""
    for line in reversed(completion_lines):
        if line.strip():
            indentation = line[:len(line) - len(line.lstrip())]
            break

    # Format the completion
    completion_text = '\n'.join(completion_lines)

    # Add bait at the end if requested
    if inject_bait:
        # Make sure to preserve indentation context
        completion_text += f"\n{indentation}{bait};"

    return completion_text


def _preprocess_code_for_javalang(code: str) -> str:
    """Comprehensive preprocessing to handle all escape sequence cases."""
    if not isinstance(code, str):
        return code

    # First normalize escaped newlines and tabs
    code = code.replace('\\n', '\n').replace('\\t', '\t')

    result = []
    i = 0
    in_string = False    # Double quote string
    in_char = False      # Character literal
    in_regex = False     # Track possible regex pattern context
    escape_active = False

    while i < len(code):
        char = code[i]
        next_char = code[i+1] if i+1 < len(code) else None

        # Track string state but only if not escaped
        if not escape_active:
            if char == '"':
                in_string = not in_string
                # Detect potential regex patterns
                if in_string and i > 0 and code[i-1] in '(=':
                    # This could be a regex pattern
                    in_regex = True
                elif not in_string:
                    in_regex = False

            elif char == "'" and not in_string:
                in_char = not in_char

        # Handle backslashes
        if char == '\\':
            if in_string or in_char:
                if next_char is None:
                    # Trailing backslash - replace with double backslash
                    result.append('\\\\')
                    i += 1
                    continue
                elif next_char in 'btnfr"\'\\':
                    # Standard Java escape sequence - keep as is
                    result.append('\\')
                    escape_active = True
                elif in_regex and next_char in 'dsSwW.()[]{}\\/+*?|^$':
                    # Regex escape - normalize by adding an extra backslash
                    result.append('\\\\')
                    escape_active = True
                else:
                    # Unknown escape - double it to be safe
                    result.append('\\\\')
                    escape_active = True
            else:
                # Backslash outside string/char - keep as is
                result.append('\\')
                escape_active = True
        else:
            # Reset escape flag and append character
            escape_active = False
            result.append(char)

        i += 1

    # Post-processing - convert problematic regex patterns
    processed = ''.join(result)

    # Replace common problematic regex patterns
    problematic_patterns = [
        (r'\\\\s+', r'\\s+'),       # Double-escaped whitespace
        (r'\\\\d+', r'\\d+'),       # Double-escaped digits
        (r'\\\\\\\.', r'\\.'),      # Triple-escaped periods
        (r'\\\\[', r'\\['),         # Double-escaped brackets
        (r'\\\\]', r'\\]')
    ]

    for bad, good in problematic_patterns:
        processed = processed.replace(bad, good)

    return processed


def create_incomplete_snippet_with_bait(complete_code, max_lines=None, lines_of_code_to_complete=None):
    """Convenience wrapper for create_incomplete_snippet with bait injection."""
    return create_incomplete_snippet(
        complete_code,
        max_lines,
        lines_of_code_to_complete,
        inject_bait=True,
        bait='pineapple.run()'
    )


def stratified_sample(df, n_samples=1000):
    """
    Sample n_samples from dataframe, preserving the distribution of user_id.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to sample from
    n_samples : int
        The number of samples to take

    Returns:
    --------
    pandas.DataFrame
        A sampled dataframe with n_samples rows
    """
    # Calculate the distribution of user_id in the original dataframe
    user_counts = df['user_id'].value_counts(normalize=True)

    # Initialize an empty dataframe to store the samples
    sampled_df = pd.DataFrame()

    # Calculate the number of samples to take from each user_id group
    # We need to handle rounding to ensure we get exactly n_samples rows
    samples_per_user = (user_counts * n_samples).astype(int)

    # If the sum is less than n_samples due to rounding,
    # add the missing samples to the most frequent users
    missing_samples = n_samples - samples_per_user.sum()
    if missing_samples > 0:
        # Get the users with the largest fractional parts
        fractional_parts = (user_counts * n_samples) - samples_per_user
        top_users = fractional_parts.sort_values(
            ascending=False).index[:missing_samples]
        for user in top_users:
            samples_per_user[user] += 1

    # Sample from each user_id group
    for user, n_samples_user in samples_per_user.items():
        if n_samples_user > 0:
            user_df = df[df['user_id'] == user]
            # If we need more samples than exist for this user, take all of them
            if n_samples_user >= len(user_df):
                user_samples = user_df
            else:
                # Otherwise take a random sample
                user_samples = user_df.sample(
                    n=n_samples_user, random_state=42)
            sampled_df = pd.concat([sampled_df, user_samples])

    return sampled_df


def stratified_sample(df, n_samples=1000):
    """
    Sample n_samples from dataframe, preserving the distribution of user_id.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to sample from
    n_samples : int
        The number of samples to take

    Returns:
    --------
    pandas.DataFrame
        A sampled dataframe with n_samples rows
    """
    # Calculate the distribution of user_id in the original dataframe
    user_counts = df['user_id'].value_counts(normalize=True)

    # Initialize an empty dataframe to store the samples
    sampled_df = pd.DataFrame()

    # Calculate the number of samples to take from each user_id group
    # We need to handle rounding to ensure we get exactly n_samples rows
    samples_per_user = (user_counts * n_samples).astype(int)

    # If the sum is less than n_samples due to rounding,
    # add the missing samples to the most frequent users
    missing_samples = n_samples - samples_per_user.sum()
    if missing_samples > 0:
        # Get the users with the largest fractional parts
        fractional_parts = (user_counts * n_samples) - samples_per_user
        top_users = fractional_parts.sort_values(
            ascending=False).index[:missing_samples]
        for user in top_users:
            samples_per_user[user] += 1

    # Sample from each user_id group
    for user, n_samples_user in samples_per_user.items():
        if n_samples_user > 0:
            user_df = df[df['user_id'] == user]
            # If we need more samples than exist for this user, take all of them
            if n_samples_user >= len(user_df):
                user_samples = user_df
            else:
                # Otherwise take a random sample
                user_samples = user_df.sample(
                    n=n_samples_user, random_state=42)
            sampled_df = pd.concat([sampled_df, user_samples])

    return sampled_df


# Example usage:
# sampled_df = stratified_sample(df, n_samples=1000)

# To verify the distribution is maintained:
def verify_distribution(original_df, sampled_df):
    """Compare original and sampled distributions of user_id"""
    original_dist = original_df['user_id'].value_counts(normalize=True)
    sampled_dist = sampled_df['user_id'].value_counts(normalize=True)

    comparison = pd.DataFrame({
        'Original %': original_dist * 100,
        'Sampled %': sampled_dist * 100,
        'Difference': (sampled_dist - original_dist) * 100
    }).round(2)

    return comparison
