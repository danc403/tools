<?php

/**
 * Extracts a substring(s) found between two specified delimiters within a larger string.
 *
 * This function can operate in two modes:
 * 1. Single Match (default): Returns the first occurrence of the content between $start and $end.
 * If no match is found, an empty string is returned.
 * 2. Multiple Matches: If $returnAll is true, an array of all found matching substrings is returned.
 * If no matches are found, an empty array is returned.
 *
 * @param string $string    The full string to search within.
 * @param string $start     The delimiter string that marks the beginning of the desired content.
 * @param string $end       The delimiter string that marks the end of the desired content.
 * @param bool   $returnAll Optional. If true, an array of all matches will be returned.
 * Defaults to false (returns only the first match as a string).
 * @return string|string[]  Returns a string if $returnAll is false (the first match).
 * Returns an array of strings if $returnAll is true (all matches).
 */
function get_string_between(string $string, string $start, string $end, bool $returnAll = false): string|array
{
    // If we're looking for a single match (default behavior)
    if (!$returnAll) {
        // Prepend a space to handle cases where $start is at the very beginning of the original string.
        // strpos returning 0 can be misinterpreted if not handled carefully.
        $stringWithSpace = " " . $string; 

        $ini = strpos($stringWithSpace, $start);

        // If the start delimiter is not found, return an empty string.
        // Using `=== false` for robust check.
        if ($ini === false) {
            return "";
        }

        // Adjust $ini to point to the beginning of the actual content after $start.
        $ini += strlen($start);    

        // Find the position of the end delimiter, starting the search from $ini.
        $lenPos = strpos($stringWithSpace, $end, $ini);

        // If the end delimiter is not found after the start delimiter, return an empty string.
        if ($lenPos === false) {
            return "";
        }

        // Calculate the length of the substring between $start and $end.
        $len = $lenPos - $ini;
        
        // Extract and return the single matching substring.
        return substr($stringWithSpace, $ini, $len);
    } 
    // If we're looking for multiple matches
    else {
        $matches = [];
        $offset = 0; // Current position in the string to start the search from

        // Loop as long as we keep finding the start delimiter
        while (($ini = strpos($string, $start, $offset)) !== false) {
            // Calculate the actual start of the content (after the $start delimiter)
            $iniActualContent = $ini + strlen($start);
            
            // Find the end delimiter, starting the search from after the start delimiter
            $endPos = strpos($string, $end, $iniActualContent);

            // If a closing tag is not found for the current opening tag, break the loop.
            // This handles malformed strings or the last opening tag without a closing tag.
            if ($endPos === false) {
                break; 
            }

            // Calculate the length of the extracted content
            $len = $endPos - $iniActualContent;
            
            // Extract the matching substring
            $extractedString = substr($string, $iniActualContent, $len);
            $matches[] = $extractedString;

            // Update the offset to start the next search *after* the current end delimiter.
            $offset = $endPos + strlen($end);
        }

        return $matches; // Return the array of all found matches
    }
}

?>
