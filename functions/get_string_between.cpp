#include "get_string_between.hpp" // Include its own header for consistency and to ensure declarations match definitions

// --- Implementation for Single Match ---
std::string getStringBetween(
    const std::string& text,
    const std::string& start,
    const std::string& end
) {
    size_t start_pos = text.find(start);
    if (start_pos == std::string::npos) {
        return "";
    }
    size_t content_start_pos = start_pos + start.length();
    size_t end_pos = text.find(end, content_start_pos);
    if (end_pos == std::string::npos) {
        return "";
    }
    size_t length = end_pos - content_start_pos;
    return text.substr(content_start_pos, length);
}

// --- Implementation for Multiple Matches ---
std::vector<std::string> getStringBetween(
    const std::string& text,
    const std::string& start,
    const std::string& end,
    bool returnAll // This parameter is for overload resolution, its value isn't used.
) {
    std::vector<std::string> matches;
    size_t offset = 0;

    while (offset < text.length()) {
        size_t start_pos = text.find(start, offset);
        if (start_pos == std::string::npos) {
            break;
        }
        size_t content_start_pos = start_pos + start.length();
        size_t end_pos = text.find(end, content_start_pos);
        if (end_pos == std::string::npos) {
            break;
        }
        size_t length = end_pos - content_start_pos;
        matches.push_back(text.substr(content_start_pos, length));
        offset = end_pos + end.length();
    }
    return matches;
}
