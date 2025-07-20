import argparse
import sys

def soundex(name: str) -> str:
    """
    Generate the Soundex code for a given string (typically a name).

    The Soundex algorithm is a phonetic algorithm for indexing names by sound,
    as pronounced in English. The goal is for homophones to be encoded to the
    same representation so that they can be matched despite minor differences
    in spelling. The resulting code is a four-character string: a letter
    followed by three numbers.

    The rules for generating a Soundex code are:
    1. Retain the first letter of the name.
    2. Replace all occurrences of the following letters with digits as shown:
       - B, F, P, V -> 1
       - C, G, J, K, Q, S, X, Z -> 2
       - D, T -> 3
       - L -> 4
       - M, N -> 5
       - R -> 6
       - A, E, I, O, U, H, W, Y -> 0 (vowels, H, W, Y are ignored after the first letter)
    3. Remove consecutive duplicate digits (e.g., "S220" becomes "S200").
    4. Remove all zeros from the code.
    5. Pad with trailing zeros and/or truncate to ensure the code is exactly
       four characters long (e.g., "S2" becomes "S200", "S2345" becomes "S234").

    Args:
        name (str): The input string to convert to a Soundex code.
                    Typically a proper noun like a person's name or a place.

    Returns:
        str: The four-character Soundex code.
    """
    if not name:
        return "0000" # Or raise an error, depending on desired behavior for empty string

    name = name.upper()
    soundex_code_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    # Step 1: Retain the first letter
    soundex_chars = [name[0]]
    # Get the code for the first letter for later comparison (Soundex Rule 3 for first letter)
    prev_code = soundex_code_map.get(name[0], '0')

    # Process remaining letters
    for char in name[1:]:
        code = soundex_code_map.get(char, '0') # Map char to digit, or '0' for vowels/ignored
        if code != '0' and code != prev_code: # Rule 2, 3: Replace, and remove consecutive duplicates
            soundex_chars.append(code)
        prev_code = code # Update prev_code for next iteration

    # Step 4: Remove all zeros (these are implicit from above, but explicit for clarity)
    # The current `soundex_chars` list already implicitly handles removing zeros
    # if they are identical to the previous code, or if they are vowels/H/W/Y.
    # We now filter out remaining '0's (from vowels/H/W/Y) that were appended
    # if they weren't consecutive duplicates of a non-zero.
    # More simply, we just want to keep the first char and then the non-zero codes
    # that are not consecutive duplicates.
    filtered_codes = [soundex_chars[0]]
    for i in range(1, len(soundex_chars)):
        if soundex_chars[i] != '0': # Keep non-zero codes
            # And also ensure no consecutive duplicates (this is partially handled above,
            # but this loop ensures it for any remaining '0's that might have broken a sequence)
            if not filtered_codes or filtered_codes[-1] != soundex_chars[i]:
                 filtered_codes.append(soundex_chars[i])


    # Re-apply the initial letter rule if it got lost in filtering
    if filtered_codes[0] != name[0]:
        filtered_codes.insert(0, name[0])
    
    # Final cleanup of duplicates if the first letter's code was same as next char's code
    # This specifically addresses cases like "Pfizer" -> P126r. If first letter code is '1' (P),
    # and next is 'f' also '1', the second '1' should be dropped.
    final_code_list = [name[0]] # Always start with the original first letter
    last_added_code = soundex_code_map.get(name[0], '0')

    for char in name[1:]:
        code = soundex_code_map.get(char, '0')
        if code != '0' and code != last_added_code:
            final_code_list.append(code)
            last_added_code = code
        # Special rule: If the original first letter's code is same as next, ignore next
        elif code != '0' and last_added_code == '0' and soundex_code_map.get(name[0], '') == code:
            # This handles cases where the first letter maps to a non-zero code,
            # and the second letter maps to the same non-zero code.
            # E.g., 'Pfizer' -> P (1) f (1) -> should be P1.
            # My current logic for `last_added_code` handles this.
            pass

    # Pad with zeros and/or truncate to exactly 4 characters
    soundex_result = "".join(final_code_list)
    
    # Pad with zeros if less than 4, truncate if more than 4
    if len(soundex_result) < 4:
        soundex_result += '0' * (4 - len(soundex_result))
    elif len(soundex_result) > 4:
        soundex_result = soundex_result[:4]

    return soundex_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the Soundex code for a given string (e.g., a name).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        required=True,
        help="The input string (e.g., 'Smith', 'Smythe', 'Robert')."
    )

    args = parser.parse_args()

    soundex_code = soundex(args.name)
    print(f"The Soundex code for '{args.name}' is: {soundex_code}")

    # Common examples for testing Soundex (from Wikipedia)
    print("\n--- Common Soundex Examples ---")
    print(f"Robert   -> {soundex('Robert')} (Expected: R163)")
    print(f"Rupert   -> {soundex('Rupert')} (Expected: R163)")
    print(f"Rubin    -> {soundex('Rubin')} (Expected: R150)")
    print(f"Ashcraft -> {soundex('Ashcraft')} (Expected: A261)")
    print(f"Aschenbrenner -> {soundex('Aschenbrenner')} (Expected: A250)") # Note: some implementations vary here
    print(f"Pfizer   -> {soundex('Pfizer')} (Expected: P126)")
    print(f"Tymczak  -> {soundex('Tymczak')} (Expected: T522)")
    print(f"Jackson  -> {soundex('Jackson')} (Expected: J250)")
