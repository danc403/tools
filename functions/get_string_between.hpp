#ifndef GET_STRING_BETWEEN_HPP
#define GET_STRING_BETWEEN_HPP

#include <string>
#include <vector>

// --- Function Overload for Single Match ---
/**
 * @brief Extracts the first substring found between two specified delimiters within a larger string.
 *
 * @param text The full string to search within. Passed by const reference for efficiency.
 * @param start The delimiter string that marks the beginning of the desired content.
 * Passed by const reference for efficiency.
 * @param end The delimiter string that marks the end of the desired content.
 * Passed by const reference for efficiency.
 * @return std::string The first matching substring. Returns an empty string if no match is found.
 */
std::string getStringBetween(
    const std::string& text,
    const std::string& start,
    const std::string& end
);

// --- Function Overload for Multiple Matches ---
/**
 * @brief Extracts all substrings found between two specified delimiters within a larger string.
 *
 * This overload is typically used by passing a 'true' value for the 'returnAll'
 * parameter, which makes the function return a vector of all found matches.
 *
 * @param text The full string to search within.
 * @param start The delimiter string that marks the beginning of the desired content.
 * @param end The delimiter string that marks the end of the desired content.
 * @param returnAll This parameter signals that multiple matches are desired. Its value is ignored.
 * @return std::vector<std::string> A vector containing all found matching substrings.
 * Returns an empty vector if no matches are found.
 */
std::vector<std::string> getStringBetween(
    const std::string& text,
    const std::string& start,
    const std::string& end,
    bool returnAll // Parameter is just for overload resolution, its value isn't used internally.
);

#endif // GET_STRING_BETWEEN_HPP
