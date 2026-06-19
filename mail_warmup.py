#!/usr/bin/env python3
import smtplib
import random
import time
import sys
import urllib.request
import xml.etree.ElementTree as ET
from email.message import EmailMessage
from bs4 import BeautifulSoup

# Configuration Parameters
SENDERS = [
    "warmup@idragonfly.net",
    "warmup@eclectacy.org"
]

RECIPIENTS = [
    "danc403@gmail.com",
    "yourtestaccount@hotmail.com"
]

# Total targeted message transmissions per twenty-four hour cycle
RUNS_PER_DAY = 3

GUARDIAN_RSS_URL = "https://www.theguardian.com/world/rss"
STALLMAN_URL = "https://stallman.org/"

def get_guardian_story():
    """Fetches a complete news story header and text block from the live RSS feed."""
    try:
        req = urllib.request.Request(
            GUARDIAN_RSS_URL, 
            headers={"User-Agent": "MailWarmupServer/3.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read()
        
        root = ET.fromstring(data)
        items = root.findall(".//item")
        if not items:
            return None, None
            
        target_item = random.choice(items)
        subject = target_item.find("title").text
        html_content = target_item.find("description").text
        
        soup = BeautifulSoup(html_content, "html.parser")
        body = soup.get_text(separator=" ")
        return subject.strip(), body.strip()
    except Exception:
        return None, None

def get_stallman_note():
    """Scrapes stallman.org and extracts an entire unedited text block note."""
    try:
        req = urllib.request.Request(
            STALLMAN_URL, 
            headers={"User-Agent": "MailWarmupServer/3.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            html_data = response.read()
            
        soup = BeautifulSoup(html_data, "html.parser")
        items = soup.find_all("li")
        valid_notes = []
        
        for item in items:
            text = item.get_text(separator=" ").strip()
            if len(text) > 80 and not text.startswith("Verify") and not text.startswith("Search"):
                valid_notes.append(text)
                
        if not valid_notes:
            return None, None
            
        full_note = random.choice(valid_notes)
        words = full_note.split()
        subject_bound = min(len(words), 6)
        subject = " ".join(words[:subject_bound]) + "..."
        return subject, full_note
    except Exception:
        return None, None

def execute_delivery_cycle():
    """Selects an unedited textual payload and routes it across the loopback interface."""
    if random.choice([True, False]):
        subject, body = get_guardian_story()
    else:
        subject, body = get_stallman_note()
        
    if not subject or not body:
        print("Network parsing temporary failure. Skipping this cycle.", flush=True)
        return

    subject = " ".join(subject.split())
    body = " ".join(body.split())
    chosen_sender = random.choice(SENDERS)
    
    print("Transmitting Payload: " + str(subject), flush=True)
    
    for recipient in RECIPIENTS:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = chosen_sender
        msg["To"] = recipient
        msg.set_content(body)
        
        try:
            with smtplib.SMTP("127.0.0.1", 25, timeout=10) as server:
                server.send_message(msg)
            print("Successfully spooled to loopback for: " + str(recipient), flush=True)
        except Exception as error:
            print("Loopback fault for target " + str(recipient) + ". Error: " + str(error), flush=True)

def run_server_engine():
    """Main execution engine looping continuously with fluid intervals."""
    print("Mail Warmup Background Server Initialized.", flush=True)
    print("Targeted frequency frequency set to: " + str(RUNS_PER_DAY) + " messages per day.", flush=True)
    
    # Calculate target math values using flat seconds allocations
    seconds_in_day = 86400
    base_interval = int(seconds_in_day / RUNS_PER_DAY)
    
    # Run the first transmission immediately upon service initialization
    execute_delivery_cycle()
    
    while True:
        # Generate a random variance factor between negative forty and positive forty percent
        variance_percentage = random.randint(-40, 40)
        modifier = int((base_interval * variance_percentage) / 100)
        current_sleep_target = base_interval + modifier
        
        print("Next transmission sequence calculated. Sleeping for " + str(current_sleep_target) + " seconds.", flush=True)
        
        # Break the long sleep into smaller blocks to ensure process responsiveness
        sleep_remaining = current_sleep_target
        while sleep_remaining > 0:
            sleep_slice = min(10, sleep_remaining)
            time.sleep(sleep_slice)
            sleep_remaining = sleep_remaining - sleep_slice
            
        execute_delivery_cycle()

if __name__ == "__main__":
    try:
        run_server_engine()
    except KeyboardInterrupt:
        print("\nServer engine shutdown cleanly via process signal.", flush=True)
        sys.exit(0)
