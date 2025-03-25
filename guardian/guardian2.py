#!python
# File: guardian.py
import requests
import xml.etree.ElementTree as ET
import sys
from bs4 import BeautifulSoup
import re
import os

# Menu data as a list of dictionaries
menu_data = [
    {"title": "World News", "url": "https://www.theguardian.com/world/rss"},
    {"title": "Top Stories(UK)", "url": "https://www.theguardian.com/uk/rss"},
    {"title": "Top Stories(US)", "url": "https://www.theguardian.com/us/rss"},
    {"title": "UK News", "url": "https://www.theguardian.com/uk-news/rss"},
    {"title": "US News", "url": "https://www.theguardian.com/us-news/rss"},
    {"title": "Business(UK)", "url": "https://www.theguardian.com/uk/business/rss"},
    {"title": "Business(US)", "url": "https://www.theguardian.com/us/business/rss"},
    {"title": "Sports(UK)", "url": "https://www.theguardian.com/uk/sport/rss"},
    {"title": "Sports(US)", "url": "https://www.theguardian.com/us/sport/rss"},
    {"title": "Culture(UK)", "url": "https://www.theguardian.com/uk/culture/rss"},
    {"title": "Culture(US)", "url": "https://www.theguardian.com/us/culture/rss"},
    {"title": "Environment(UK)", "url": "https://www.theguardian.com/uk/environment/rss"},
    {"title": "Environment(US)", "url": "https://www.theguardian.com/us/environment/rss"},
    {"title": "Technology(UK)", "url": "https://www.theguardian.com/uk/technology/rss"},
    {"title": "Technology(US)", "url": "https://www.theguardian.com/us/technology/rss"},
]

# Create a dictionary for easy access
menu = {item['title']: item['url'] for item in menu_data}


# Function to fetch and parse the XML feed
def fetch_and_parse_feed(feed_url):
    try:
        response = requests.get(feed_url)
        response.raise_for_status()  # Raise an error for bad responses
        xml_content = response.content.decode('utf-8')
        root = ET.fromstring(xml_content)
        return root
    except requests.exceptions.RequestException as e:
        print(f"Error fetching feed: {e}")
        return None  # Return None to indicate failure
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None  # Return None to indicate failure


# Function to extract information from an item
def extract_item_info(item, namespaces):
    title = item.find('title').text if item.find('title') is not None else ''
    link = item.find('link').text if item.find('link') is not None else ''
    description = item.find('description').text if item.find('description') is not None else ''

    # Namespace handling: Check if namespaces is not None before accessing
    creator = item.find('dc:creator', namespaces)
    creator_text = creator.text if creator is not None else ''

    encoded_content = item.find('content:encoded', namespaces)
    encoded_content_text = encoded_content.text if encoded_content is not None else ''

    return {
        'title': title,
        'link': link,
        'description': description,
        'creator': creator_text,
        'content': encoded_content_text
    }

def strip_tags(html):
    """Strip HTML tags from a string, but preserve links."""
    soup = BeautifulSoup(html, 'html.parser')
    output = ''
    for element in soup.recursiveChildGenerator():
        if isinstance(element, str):
            output += element
        elif element.name == 'a':
            output += f"{element.text} ({element['href']}) "  # Display link text and URL

    return output

def fetch_and_extract_article(url):
    """Fetches the article from the URL and extracts the text content."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Attempt to extract the main article content.  Adjust this based on the target website's structure.
        article = soup.find('div', class_='article') or soup.find('article') or soup.find('div', id='content') or soup  # Example selectors - adjust as needed

        if article:
            text = article.get_text(separator='\n', strip=True)  # Get text with line breaks
            return text
        else:
            return "Could not extract article content."

    except requests.exceptions.RequestException as e:
        return f"Error fetching article: {e}"
    except Exception as e:
        return f"Error processing article: {e}"


def extract_links_from_description(description):
    """Extracts and numbers links from the description."""
    soup = BeautifulSoup(description, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', href=True):
        links.append(a_tag['href'])
    return links

def display_item_list(items, read_items, start_index=0, items_per_page=10):
    """Displays the list of item titles with numbers for selection, marking read items, with pagination."""
    end_index = min(start_index + items_per_page, len(items))
    print("\nAvailable Articles:")
    for i in range(start_index, end_index):
        item = items[i]
        read_marker = "[READ]" if i in read_items else ""
        print(f"{i+1}. {item['title']} {read_marker}")

    print("\nOptions:")
    if end_index < len(items):
        print("n. Next Page")
    if start_index > 0:
        print("p. Previous Page")
    print("m. Main Menu")
    print("q. Exit")

    return start_index, end_index  # Return indices for pagination

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')



def display_article_text(text):
    """Displays the extracted article text."""
    print("\n--- Article Content ---")
    print(text)
    print("\n--- End of Article ---")

def save_article_to_file(text, filename="article.txt"):
    """Saves the article text to a file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nArticle saved to '{filename}'")
    except Exception as e:
        print(f"Error saving article to file: {e}")



def display_item_details(item, item_index, items, read_items, last_location):
    """Displays the details of a selected item and provides options to read the linked article."""
    links_in_description = extract_links_from_description(item['description'])
    link_count = len(links_in_description) #Count of the links in the description.
    while True:
        print("--------------------------------------")
        for key, value in item.items():
            if key == 'description' or key == 'content':
                print(f"{key}: {strip_tags(str(value))}")  # Strip HTML tags for cleaner display, preserving links
            else:
                print(f"{key}: {str(value)}")
        print("--------------------------------------")

        if links_in_description:
            print("\nLinks in Description:")
            for i, link in enumerate(links_in_description):
                print(f"{i + 1}. {link}")

        print("\nOptions:")
        if links_in_description:
            print(f"Enter 1-{link_count} to read the corresponding linked page.")
        print("s. Save article to file (article.txt)") # Save option
        print("b. Back")
        print("m. Main Menu")
        print("q. Exit")

        choice = input("Enter your choice: ").lower()

        if links_in_description and choice.isdigit() and 1 <= int(choice) <= len(links_in_description):
            link_index = int(choice) - 1
            article_text = fetch_and_extract_article(links_in_description[link_index])
            display_article_text(article_text)
            current_article_text = article_text  #Store the article text for potential saving

            while True:
                save_or_back_choice = input("Enter 's' to save the article to file, 'b' to go back to the article details, 'm' for main menu, or 'q' to exit: ").lower()
                if save_or_back_choice == 'b':
                    break  # Back to item details
                elif save_or_back_choice == 'm':
                    return 'main_menu'  # Signal to return to the main menu
                elif save_or_back_choice == 'q':
                    print("Exiting...")
                    sys.exit(0)
                elif save_or_back_choice == 's':
                    save_article_to_file(article_text) #Save the article
                    break
                else:
                    print("Invalid choice.")

        elif choice == 's': # Save item content
            save_article_to_file(strip_tags(item['content'])) #Strip tags first.
        elif choice == 'b':
            return last_location  #Signal to go back to last location.
        elif choice == 'm':
            return 'main_menu'  # Signal to return to the main menu
        elif choice == 'q':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice.")

def display_menu():
    """Displays the main menu options."""
    print("\nMain Menu:")
    for i, (menu_item, _) in enumerate(menu.items()):
        print(f"{i+1}. {menu_item}")
    print("q. Exit")


def main():
    """Main function to drive the RSS feed interaction."""
    read_items = set() # Keep track of read item indices
    last_location = 'main_menu' #Start on the main menu.
    current_feed_key = None # Store the current feed key
    start_index = 0  # Start index for pagination
    items_per_page = 10  # Number of items to display per page
    items = [] #Store the items here

    while True:
        clear_screen() #Clear screen at top of each loop.

        if last_location == 'main_menu':
            display_menu()
            choice = input("Enter the number of the feed to view (m for Main Menu, q to exit): ").lower()

            if choice == 'q':
                print("Exiting...")
                break

            try:
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(menu):
                    current_feed_key = list(menu.keys())[choice_index]
                    feed_url = menu[current_feed_key]

                    root = fetch_and_parse_feed(feed_url)

                    if root is None:
                        last_location = 'main_menu'
                        continue

                    namespaces = {'dc': 'http://purl.org/dc/elements/1.1/',
                                  'content': 'http://purl.org/rss/1.0/modules/content/'}
                    items_elements = root.findall('.//item')
                    items = [extract_item_info(item, namespaces) for item in items_elements]  # Extract and store item data
                    start_index = 0 #Reset pagination
                    last_location = current_feed_key # Set last location to the feed
                else:
                    print("Invalid choice. Please enter a number from the list.")
                    last_location = 'main_menu'

            except ValueError:
                print("Invalid input. Please enter a number.")
                last_location = 'main_menu'

        else: #Viewing a feed
            #No need to fetch the feed again, use the items already fetched.
            display_item_list(items, read_items, start_index, items_per_page)
            choice = input("Enter the number of the article to view (n for Next, p for Previous, m for Main Menu, q to exit): ").lower()


            if choice == 'q':
                print("Exiting...")
                break
            elif choice == 'm':
                last_location = 'main_menu' # Return to main menu
                continue
            elif choice == 'n':
                start_index += items_per_page
                if start_index >= len(items):
                    start_index -= items_per_page #Avoid going past the end
                continue # Redisplay the list
            elif choice == 'p':
                start_index -= items_per_page
                if start_index < 0:
                    start_index = 0 # Don't go negative
                continue #Redisplay the list



            try:
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(items):
                    #Display Item Details
                    result = display_item_details(items[choice_index], choice_index, items, read_items, last_location)
                    if result == 'main_menu':
                         last_location = 'main_menu'
                    else:
                        last_location = result #Update to last location

                    read_items.add(choice_index)  # Mark as read if user viewed or linked page.
                else:
                    print("Invalid choice. Please enter a number from the list.")


            except ValueError:
                print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    main()
