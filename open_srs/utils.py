import argparse
import json
import os
import sys

config_file_path = "config.json"
try:
    with open(config_file_path, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Configuration file '{config_file_path}' not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in '{config_file_path}'.")
    sys.exit(1)

# Ensure the necessary keys are present in the config
required_keys = [
    'live_reseller_username', 'live_api_key', 'live_api_host_port',
    'test_reseller_username', 'test_api_key', 'test_api_host_port'
]

for key in required_keys:
    if key not in config:
        print(f"Error: Missing required key '{key}' in configuration.")
        sys.exit(1)

connection_options = {
    'live': {
        'reseller_username': config['live_reseller_username'],
        'api_key': config['live_api_key'],
        'api_host_port': config['live_api_host_port'],
    },
    'test': {
        'reseller_username': config['test_reseller_username'],
        'api_key': config['test_api_key'],
        'api_host_port': config['test_api_host_port'],
    }
}

# 0 for live, 1 for test
TEST_MODE = 0

if TEST_MODE == 1:
    connection_details = connection_options['test']
else:
    connection_details = connection_options['live']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Interact with OpenSRS API for various domain management tasks.')
    
    subparsers = parser.add_subparsers(dest='command', help='Domain management commands', required=True)
    
    # Get balance
    parser_get_balance = subparsers.add_parser('balance', help='Get account balance')
    
    # Check domain belongs to RSP
    parser_check_domain_belongs = subparsers.add_parser('check-domain', help='Check if a domain belongs to the RSP')
    parser_check_domain_belongs.add_argument('domain', type=str, help='The domain to check')
    
    # Check domain availability
    parser_check_availability = subparsers.add_parser('check-avail', help='Check domain availability')
    parser_check_availability.add_argument('domain', type=str, help='The domain to check availability for')
    
    # Send password reset email
    parser_send_password = subparsers.add_parser('send-password', help='Send password reset email')
    parser_send_password.add_argument('domain_name', type=str, help='The domain name for which the password is sent')
    parser_send_password.add_argument('send_to', type=str, help='The contact to which the password is to be sent (owner or admin)', choices=['owner', 'admin'])
    parser_send_password.add_argument('--sub_user', type=int, help='Send password to sub-user (0 or 1)', choices=[0, 1], default=0)
    
    # Get price
    parser_get_price = subparsers.add_parser('get-price', help='Get domain price')
    parser_get_price.add_argument('domain', type=str, help='The domain to check the price for')
    parser_get_price.add_argument('-p', '--period', type=int, help='The desired registration period in years', default=1)
    parser_get_price.add_argument('-a', '--all_periods', action='store_true', help='Return prices for all available periods')
    
    # Redeem domain
    parser_redeem_domain = subparsers.add_parser('redeem', help='Redeem a domain')
    parser_redeem_domain.add_argument('domain', type=str, help='The domain to redeem')
    parser_redeem_domain.add_argument('--registrant_ip', type=str, help='The valid IP address of the registrant (optional)', default=None)
    
    # Send registrant verification email
    parser_send_registrant_verification_email = subparsers.add_parser('send-verification', help='Send registrant verification email')
    parser_send_registrant_verification_email.add_argument('domain', type=str, help='The domain to send the verification email for')
    
    # SW Register
    parser_sw_register = subparsers.add_parser('register', help='Register or transfer a domain')
    parser_sw_register.add_argument('domain', type=str, help='The domain name to be registered or transferred')
    parser_sw_register.add_argument('reg_type', type=str, help='The type of registration (new, transfer, landrush, sunrise)', choices=['new', 'transfer', 'landrush', 'sunrise'])
    parser_sw_register.add_argument('period', type=int, help='The registration period in years (1-10)')
    parser_sw_register.add_argument('reg_username', type=str, help='The username of the registrant')
    parser_sw_register.add_argument('reg_password', type=str, help='The password of the registrant')
    parser_sw_register.add_argument('contact_set', type=str, help='JSON string containing contact information for owner, admin, billing, and tech contacts')
    parser_sw_register.add_argument('nameserver_list', type=str, help='JSON string containing a list of nameservers with their sort order')
    parser_sw_register.add_argument('--handle', type=str, help='Indicates how to process the order (process or save)', choices=['process', 'save'], default='process')
    parser_sw_register.add_argument('--custom_nameservers', type=int, help='Use custom nameservers (0 or 1)', choices=[0, 1], default=0)
    parser_sw_register.add_argument('--custom_transfer_nameservers', type=int, help='Use custom nameservers for transfers (0 or 1)', choices=[0, 1], default=0)
    parser_sw_register.add_argument('--custom_tech_contact', type=int, help='Use custom tech contact (0 or 1)', choices=[0, 1], default=0)
    parser_sw_register.add_argument('--f_lock_domain', type=int, help='Lock the domain (0 or 1)', choices=[0, 1], default=0)
    parser_sw_register.add_argument('--f_parkp', type=str, help='Enable Parked Pages (Y or N)', choices=['Y', 'N'], default='N')
    parser_sw_register.add_argument('--f_whois_privacy', type=int, help='Enable WHOIS Privacy (0 or 1)', choices=[0, 1], default=0)
    parser_sw_register.add_argument('--affiliate_id', type=str, help='Affiliate ID for tracking orders')
    parser_sw_register.add_argument('--auto_renew', type=int, help='Set domain to auto-renew (0 or 1)', choices=[0, 1], default=0)
    parser_sw_register.add_argument('--auth_info', type=str, help='Transfer authcode for the domain')
    parser_sw_register.add_argument('--change_contact', type=int, help='Change contact information during transfer (0 or 1)', choices=[0, 1], default=0)
    parser_sw_register.add_argument('--comments', type=str, help='Additional notes for the order')
    parser_sw_register.add_argument('--dns_template', type=str, help='DNS template to use for the domain')
    parser_sw_register.add_argument('--encoding_type', type=str, help='Encoding type for ID')
    parser_sw_register.add_argument('--intended_use', type=str, help='Intended use for .scot registrations')
    parser_sw_register.add_argument('--link_domains', type=int, help='Link domains (0 or 1)', choices=[0, 1], default=0)
    parser_sw_register.add_argument('--master_order_id', type=str, help='Master order ID for linked domains')
    parser_sw_register.add_argument('--owner_confirm_address', type=str, help='Email address for transfer confirmation')
    parser_sw_register.add_argument('--messaging_language', type=str, help='Messaging language for customer notifications')
    parser_sw_register.add_argument('--tld_data', type=str, help='TLD-specific data')

    # Renew domain
    parser_renew_domain = subparsers.add_parser('renew', help='Renew a domain')
    parser_renew_domain.add_argument('domain', type=str, help='The domain to renew')
    parser_renew_domain.add_argument('-y', '--currentexpirationyear', type=int, help='The domain\'s current expiration year')
    parser_renew_domain.add_argument('-p', '--period', type=int, help='The renewal period, from 1 to 10 years', choices=range(1, 11), default=1)
    parser_renew_domain.add_argument('--auto_renew', type=int, help='Indicates whether the domain should be set to auto-renew (0 or 1)', choices=[0, 1])
    parser_renew_domain.add_argument('--affiliate_id', type=str, help='ID for tracking orders via affiliates', default=None)
    parser_renew_domain.add_argument('-i', '--registrant_ip', type=str, help='The valid IP address of the registrant (optional)', default=None)
    parser_renew_domain.add_argument('--f_parkp', type=str, help='Enables the Parked Pages Program (Y or N)', choices=['Y', 'N'], default=None)
    
    # Revoke domain
    parser_revoke_domain = subparsers.add_parser('revoke', help='Revoke a domain')
    parser_revoke_domain.add_argument('domain', type=str, help='The domain to revoke')
    parser_revoke_domain.add_argument('--notes', type=str, help='Information relevant to the action (optional)', default=None)
    
    # Set domain affiliate ID
    parser_set_domain_affiliate_id = subparsers.add_parser('set-affiliate', help='Set domain affiliate ID')
    parser_set_domain_affiliate_id.add_argument('domain', type=str, help='The domain to assign the affiliate ID to')
    parser_set_domain_affiliate_id.add_argument('affiliate_id', type=str, help='The affiliate ID to assign (max 256 characters)')
    
    args = parser.parse_args()
    return args
