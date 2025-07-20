import typing

def get_string_between(text: str, start: str, end: str, return_all: bool = False) -> typing.Union[str, typing.List[str]]:
    """
    Extracts a substring(s) found between two specified delimiters within a larger string.

    This function can operate in two modes:
    1. Single Match (default): Returns the first occurrence of the content between 'start' and 'end'.
       If no match is found, an empty string is returned.
    2. Multiple Matches: If 'return_all' is True, a list of all found matching substrings is returned.
       If no matches are found, an empty list is returned.

    Args:
        text (str): The full string to search within.
        start (str): The delimiter string that marks the beginning of the desired content.
        end (str): The delimiter string that marks the end of the desired content.
        return_all (bool, optional): If True, a list of all matches will be returned.
                                     Defaults to False (returns only the first match as a string).

    Returns:
        Union[str, List[str]]: Returns a string if 'return_all' is False (the first match).
                               Returns a list of strings if 'return_all' is True (all matches).
    """
    if not return_all:
        # Single match mode
        # The PHP ' " ".$string ' trick is not needed in Python's find/index,
        # as it correctly returns 0 for a match at the beginning.
        start_index = text.find(start)

        # If the start delimiter is not found, return an empty string.
        if start_index == -1:
            return ""

        # Adjust start_index to point to the beginning of the actual content after 'start'.
        content_start_index = start_index + len(start)

        # Find the position of the end delimiter, starting the search from content_start_index.
        end_index = text.find(end, content_start_index)

        # If the end delimiter is not found after the start delimiter, return an empty string.
        if end_index == -1:
            return ""

        # Extract and return the single matching substring.
        return text[content_start_index:end_index]
    else:
        # Multiple matches mode
        matches = []
        offset = 0  # Current position in the string to start the search from

        # Loop as long as we keep finding the start delimiter
        while True:
            start_index = text.find(start, offset)

            if start_index == -1:
                # No more start delimiters found
                break

            # Calculate the actual start of the content (after the 'start' delimiter)
            content_start_index = start_index + len(start)

            # Find the end delimiter, starting the search from after the start delimiter
            end_index = text.find(end, content_start_index)

            if end_index == -1:
                # No closing tag found for the current opening tag, break.
                # This handles malformed strings or the last opening tag without a closing tag.
                break

            # Extract the matching substring
            extracted_string = text[content_start_index:end_index]
            matches.append(extracted_string)

            # Update the offset to start the next search *after* the current end delimiter.
            offset = end_index + len(end)

        return matches

