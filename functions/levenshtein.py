import argparse

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    The Levenshtein distance, also known as edit distance, quantifies the
    minimum number of single-character edits (insertions, deletions, or
    substitutions) required to change one string into the other. This
    implementation uses the dynamic programming approach, which is efficient
    for calculating distances between shorter strings.

    Args:
        s1 (str): The first string for comparison.
        s2 (str): The second string for comparison.

    Returns:
        int: The Levenshtein distance (an integer representing the minimum
             number of edits). Returns 0 if strings are identical.
    """
    m, n = len(s1), len(s2)

    # Create a 2D dynamic programming table (matrix)
    # dp[i][j] will store the Levenshtein distance between
    # the first i characters of s1 and the first j characters of s2.
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and first column
    # dp[i][0] represents the distance when s2 is an empty string (i deletions needed)
    for i in range(m + 1):
        dp[i][0] = i
    # dp[0][j] represents the distance when s1 is an empty string (j insertions needed)
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table using the recurrence relation
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Cost of substitution: 0 if characters match, 1 otherwise
            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            # The current cell dp[i][j] is the minimum of:
            # 1. Deletion: dp[i-1][j] + 1 (delete character s1[i-1])
            # 2. Insertion: dp[i][j-1] + 1 (insert character s2[j-1])
            # 3. Substitution/Match: dp[i-1][j-1] + cost (substitute s1[i-1] with s2[j-1] or match them)
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    # The bottom-right cell contains the Levenshtein distance for the full strings
    return dp[m][n]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the Levenshtein (edit) distance between two strings.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--string1', '-s1',
        type=str,
        required=True,
        help="The first string for comparison (e.g., 'kitten')."
    )
    parser.add_argument(
        '--string2', '-s2',
        type=str,
        required=True,
        help="The second string for comparison (e.g., 'sitting')."
    )

    args = parser.parse_args()

    distance = levenshtein_distance(args.string1, args.string2)
    print(f"The Levenshtein distance between '{args.string1}' and '{args.string2}' is: {distance}")
