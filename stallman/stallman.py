#!/usr/bin/env python3

import requests
import xml.etree.ElementTree as ET
import sys
from bs4 import BeautifulSoup
import re

feed = "https://stallman.org/rss/rss.xml"

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
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        sys.exit(1)


# Function to extract information from an item
def extract_item_info(item, namespaces):
    title = item.find('title').text if item.find('title') is not None else ''
    link = item.find('link').text if item.find('link') is not None else ''
    description = item.find('description').text if item.find('description') is not None else ''

    creator = item.find('dc:creator', namespaces).text if item.find('dc:creator', namespaces) is not None else ''
    encoded_content = item.find('content:encoded', namespaces).text if item.find('content:encoded', namespaces) is not None else '' # Get full content
    return {
        'title': title,
        'link': link,
        'description': description,
        'creator': creator,
        'content': encoded_content
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

def display_item_list(items, read_items):
    """Displays the list of item titles with numbers for selection, marking read items."""
    print("\nAvailable Articles:")
    for i, item in enumerate(items):
        read_marker = "[READ]" if i in read_items else ""
        print(f"{i+1}. {item['title']} {read_marker}")
    print("m. Main Menu")
    print("q. Exit")

def display_article_text(text):
    """Displays the extracted article text."""
    print("\n--- Article Content ---")
    print(text)
    print("\n--- End of Article ---")


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
        print("b. Back")
        print("m. Main Menu")
        print("q. Exit")

        choice = input("Enter your choice: ").lower()

        if links_in_description and choice.isdigit() and 1 <= int(choice) <= len(links_in_description):
            link_index = int(choice) - 1
            article_text = fetch_and_extract_article(links_in_description[link_index])
            display_article_text(article_text)

            while True:
                back_choice = input("Enter 'b' to go back to the article details, 'm' for main menu, or 'q' to exit: ").lower()
                if back_choice == 'b':
                    break  # Back to item details
                elif back_choice == 'm':
                    return 'main_menu'  # Signal to return to the main menu
                elif back_choice == 'q':
                    print("Exiting...")
                    sys.exit(0)
                else:
                    print("Invalid choice.")



        elif choice == 'b':
            return last_location  #Signal to go back to last location.
        elif choice == 'm':
            return 'main_menu'  # Signal to return to the main menu
        elif choice == 'q':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice.")


def main():
    """Main function to drive the RSS feed interaction."""
    root = fetch_and_parse_feed(feed)
    namespaces = {'dc': 'http://purl.org/dc/elements/1.1/',
                  'content': 'http://purl.org/rss/1.0/modules/content/'}
    items_elements = root.findall('.//item')
    items = [extract_item_info(item, namespaces) for item in items_elements]  # Extract and store item data
    read_items = set() # Keep track of read item indices
    last_location = 'main_menu' #Start on the main menu.


    while True:
        if last_location == 'main_menu':
            display_item_list(items, read_items)
            choice = input("Enter the number of the article to view (m for Main Menu, q to exit): ").lower()
        else:
             choice = last_location


        if choice == 'q':
            print("Exiting...")
            break
        elif choice == 'm':
            last_location = 'main_menu' # Return to main menu (do nothing, loop restarts)
            continue


        try:
            if choice != 'main_menu':
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
            else:
                last_location = 'main_menu'
        except ValueError:
            print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    main()
