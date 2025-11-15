# File: reddit_feed.py
import requests
import xml.etree.ElementTree as ET
import sys
from bs4 import BeautifulSoup
import os

# Menu data for specific subreddits, sorted newest first
menu_data = {
    "r/blind (new)": "https://www.reddit.com/r/blind/new/.rss",
    "r/localllama (new)": "https://www.reddit.com/r/localllama/new/.rss",
    "r/printsf (new)": "https://www.reddit.com/r/printsf/new/.rss",
    "Enter a custom subreddit...": None
}

# --- Core Functions for RSS/Atom Parsing ---

def fetch_and_parse_feed(feed_url):
    """
    Fetches and parses an Atom feed from a given URL.
    Returns the root element or None on failure.
    """
    try:
        response = requests.get(feed_url, headers={'User-agent': 'MyRedditBot 1.0'})
        response.raise_for_status()
        xml_content = response.content.decode('utf-8')
        root = ET.fromstring(xml_content)
        return root
    except requests.exceptions.RequestException as e:
        print(f"Error fetching feed: {e}")
        return None
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None

def extract_post_info(entry):
    """Extracts information from an Atom <entry> element (a post)."""
    # Atom feeds use namespaces, which need to be handled.
    namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
    
    title = entry.find('atom:title', namespaces).text if entry.find('atom:title', namespaces) is not None else ''
    link = entry.find('atom:link', namespaces).attrib['href'] if entry.find('atom:link', namespaces) is not None else ''
    author = entry.find('atom:author/atom:name', namespaces).text if entry.find('atom:author/atom:name', namespaces) is not None else ''
    content = entry.find('atom:content', namespaces).text if entry.find('atom:content', namespaces) is not None else ''
    
    return {
        'title': title,
        'link': link,
        'author': author,
        'content': content
    }

def extract_comment_info(entry):
    """Extracts information from an Atom <entry> element (a comment)."""
    namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
    
    author = entry.find('atom:author/atom:name', namespaces).text if entry.find('atom:author/atom:name', namespaces) is not None else ''
    content = entry.find('atom:content', namespaces).text if entry.find('atom:content', namespaces) is not None else ''
    
    return {
        'author': author,
        'content': content
    }

def strip_html_tags(html):
    """Strips HTML tags from a string."""
    if not html:
        return ""
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator='\n')

# --- Display and User Interaction Functions ---

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_main_menu():
    """Displays the main menu with predefined and custom subreddit options."""
    print("\n--- Reddit Feed Menu ---")
    for i, title in enumerate(menu_data.keys()):
        print(f"{i+1}. {title}")
    print("q. Exit")

def display_post_list(posts, start_index=0, items_per_page=10):
    """Displays the list of post titles with numbers for selection, with pagination."""
    end_index = min(start_index + items_per_page, len(posts))
    print("\n--- Available Posts ---")
    for i in range(start_index, end_index):
        post = posts[i]
        print(f"{i+1}. {post['title']} by {post['author']}")

    print("\nOptions:")
    if end_index < len(posts):
        print("n. Next Page")
    if start_index > 0:
        print("p. Previous Page")
    print("m. Main Menu")
    print("q. Exit")
    
    return start_index, end_index

def display_comments(post_link):
    """
    Fetches the comment feed for a post and displays the comments.
    """
    # The comments RSS link is the post URL plus .rss
    comments_url = f"{post_link}.rss"
    root = fetch_and_parse_feed(comments_url)

    if root is None:
        print("Failed to fetch comments.")
        return

    # Check for posts and comments
    namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
    comment_entries = root.findall('.//atom:entry', namespaces)

    if not comment_entries:
        print("\nNo comments found for this post.")
        return

    print("\n--- Comments ---")
    for entry in comment_entries:
        comment_info = extract_comment_info(entry)
        author = comment_info['author']
        content = strip_html_tags(comment_info['content'])
        print(f"\nAuthor: {author}\n{content}")
        print("---")
        
    input("\nPress Enter to return to post details...")


def display_post_details(post):
    """Displays the details of a selected post and provides options to view comments."""
    while True:
        print("\n--- Post Details ---")
        print(f"Title: {post['title']}")
        print(f"Author: {post['author']}")
        print(f"Link: {post['link']}")
        print("\nContent:")
        print(strip_html_tags(post['content']))
        print("--------------------")

        print("\nOptions:")
        print("c. View comments")
        print("b. Back to post list")
        print("m. Main Menu")
        print("q. Exit")

        choice = input("Enter your choice: ").lower()

        if choice == 'c':
            display_comments(post['link'])
            # After viewing comments, return to post details menu
        elif choice == 'b':
            return 'post_list'
        elif choice == 'm':
            return 'main_menu'
        elif choice == 'q':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice.")

# --- Main Logic ---

def main():
    """Main function to drive the Reddit feed interaction."""
    last_location = 'main_menu'
    current_subreddit_url = None
    posts = []
    start_index = 0
    items_per_page = 10

    while True:
        clear_screen()
        
        if last_location == 'main_menu':
            display_main_menu()
            choice = input("Enter your choice: ").lower()

            if choice == 'q':
                print("Exiting...")
                break

            try:
                if choice.isdigit():
                    choice_index = int(choice) - 1
                    menu_keys = list(menu_data.keys())
                    if 0 <= choice_index < len(menu_keys):
                        selected_option = menu_keys[choice_index]
                        if menu_data[selected_option]:
                            current_subreddit_url = menu_data[selected_option]
                            last_location = 'post_list'
                        else: # Custom subreddit option
                            custom_subreddit = input("Enter the subreddit name (e.g., 'python'): ")
                            current_subreddit_url = f"https://www.reddit.com/r/{custom_subreddit}/new/.rss"
                            last_location = 'post_list'
                    else:
                        print("Invalid choice. Please enter a number from the list.")
                        last_location = 'main_menu'
                        input("Press Enter to continue...")
                else:
                    print("Invalid input. Please enter a number or 'q'.")
                    last_location = 'main_menu'
                    input("Press Enter to continue...")

            except ValueError:
                print("Invalid input.")
                last_location = 'main_menu'
                input("Press Enter to continue...")

        elif last_location == 'post_list':
            # Fetch posts if we don't have them yet
            if not posts:
                root = fetch_and_parse_feed(current_subreddit_url)
                if root is None:
                    posts = []
                    last_location = 'main_menu'
                    input("Press Enter to continue...")
                    continue
                
                # Atom feeds use 'entry' instead of 'item'
                namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
                post_entries = root.findall('.//atom:entry', namespaces)
                posts = [extract_post_info(entry) for entry in post_entries]
                
                if not posts:
                    print("\nNo posts found for this subreddit.")
                    last_location = 'main_menu'
                    input("Press Enter to continue...")
                    continue

            display_post_list(posts, start_index, items_per_page)
            choice = input("Enter your choice: ").lower()

            if choice == 'q':
                print("Exiting...")
                break
            elif choice == 'm':
                last_location = 'main_menu'
                posts = [] # Clear posts for the next selection
                continue
            elif choice == 'n':
                start_index += items_per_page
                if start_index >= len(posts):
                    start_index -= items_per_page
                continue
            elif choice == 'p':
                start_index -= items_per_page
                if start_index < 0:
                    start_index = 0
                continue

            try:
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(posts):
                    # Display Post Details and Comments
                    result = display_post_details(posts[choice_index])
                    last_location = result
                else:
                    print("Invalid choice. Please enter a number from the list.")
                    input("Press Enter to continue...")
            except ValueError:
                print("Invalid input.")
                input("Press Enter to continue...")

if __name__ == "__main__":
    main()
