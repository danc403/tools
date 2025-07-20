#include "get_string_between.hpp" // Include our reusable string utility functions

#include <iostream>  // For std::cout, std::cerr, std::endl
#include <string>    // For std::string
#include <vector>    // For std::vector
#include <fstream>   // For std::ifstream (file reading)
#include <sstream>   // For std::stringstream (buffer for file/stdin)
#include <stdexcept> // For std::runtime_error (custom exceptions)

// Helper function to read entire file contents into a string
std::string readFileContents(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file '" + filePath + "'");
    }
    std::stringstream buffer;
    buffer << file.rdbuf(); // Read the entire file buffer into the stringstream
    return buffer.str();
}

// Helper function to read entire standard input into a string
std::string readStdinContents() {
    std::string line;
    std::string content;
    // Read line by line from stdin until EOF
    while (std::getline(std::cin, line)) {
        content += line + "\n";
    }
    // Remove the very last newline if input was not empty, for consistency with file reading.
    if (!content.empty() && content.back() == '\n') {
        content.pop_back();
    }
    return content;
}

// Function to print application usage help
void printHelp() {
    std::cout << "Usage: getstringbetween -s <START_DELIMITER> -e <END_DELIMITER> [OPTIONS]\n"
              << "Extracts substrings found between specified delimiters.\n\n"
              << "Required arguments:\n"
              << "  -s <START_DELIMITER>  The string marking the beginning of the desired content.\n"
              << "  -e <END_DELIMITER>    The string marking the end of the desired content.\n\n"
              << "Input source (choose one or use stdin if none specified):\n"
              << "  -f <FILE_PATH>        Read content from the specified file.\n"
              << "  -c <CONTENT_STRING>   Use the provided string literal as direct input.\n\n"
              << "Optional arguments:\n"
              << "  -a, --all             Return all matching substrings (default: only the first).\n"
              << "  -h, --help            Display this help message and exit.\n";
}

int main(int argc, char* argv[]) {
    std::string start_delimiter;
    std::string end_delimiter;
    std::string input_source_value; // Stores either file path or direct content string
    bool read_from_file_flag = false;
    bool read_from_arg_string_flag = false;
    bool return_all_matches_flag = false;

    // --- Command-line argument parsing loop ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-s") {
            if (i + 1 < argc) { // Check if there's a next argument for the value
                start_delimiter = argv[++i];
            } else {
                std::cerr << "Error: -s requires an argument.\n";
                printHelp();
                return 1; // Exit with error
            }
        } else if (arg == "-e") {
            if (i + 1 < argc) {
                end_delimiter = argv[++i];
            } else {
                std::cerr << "Error: -e requires an argument.\n";
                printHelp();
                return 1;
            }
        } else if (arg == "-f") {
            if (i + 1 < argc) {
                input_source_value = argv[++i];
                read_from_file_flag = true;
            } else {
                std::cerr << "Error: -f requires a file path.\n";
                printHelp();
                return 1;
            }
        } else if (arg == "-c") {
            if (i + 1 < argc) {
                input_source_value = argv[++i];
                read_from_arg_string_flag = true;
            } else {
                std::cerr << "Error: -c requires a string argument.\n";
                printHelp();
                return 1;
            }
        } else if (arg == "-a" || arg == "--all") {
            return_all_matches_flag = true;
        } else if (arg == "-h" || arg == "--help") {
            printHelp();
            return 0; // Exit successfully after showing help
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            printHelp();
            return 1;
        }
    }

    // --- Input Validation ---
    if (start_delimiter.empty() || end_delimiter.empty()) {
        std::cerr << "Error: -s (start delimiter) and -e (end delimiter) are required.\n";
        printHelp();
        return 1;
    }

    // --- Read Input Content ---
    std::string text_to_search;
    try {
        if (read_from_file_flag) {
            text_to_search = readFileContents(input_source_value);
        } else if (read_from_arg_string_flag) {
            text_to_search = input_source_value;
        } else {
            // Default behavior: if no -f or -c, read from standard input (stdin)
            text_to_search = readStdinContents();
        }
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1; // Exit on file reading error
    }

    // --- Perform String Extraction and Print Results ---
    if (return_all_matches_flag) {
        // Call the overload for multiple matches
        std::vector<std::string> results = getStringBetween(text_to_search, start_delimiter, end_delimiter, true);
        if (results.empty()) {
            // You can choose to print nothing or a specific message here
            // std::cout << "No matches found.\n";
        } else {
            for (const auto& match : results) {
                std::cout << match << "\n---\n"; // Separate multiple results with '---'
            }
        }
    } else {
        // Call the overload for a single match
        std::string result = getStringBetween(text_to_search, start_delimiter, end_delimiter);
        std::cout << result << std::endl;
    }

    return 0; // Successful application exit
}
