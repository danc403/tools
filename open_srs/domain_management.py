import requests
import hashlib
import xml.etree.ElementTree as ET
from utils import connection_details

def get_balance():
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="object">DOMAIN</item>
            <item key="action">GET_BALANCE</item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    #print(xml)

    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            root = ET.fromstring(response_text)
            is_success_item = root.find(".//item[@key='is_success']")
            response_code_item = root.find(".//item[@key='response_code']")
            response_text_item = root.find(".//item[@key='response_text']")
            balance_item = root.find(".//item[@key='balance']")
            hold_balance_item = root.find(".//item[@key='hold_balance']")

            if is_success_item is not None and is_success_item.text == '1':
                balance = balance_item.text if balance_item is not None else 'N/A'
                hold_balance = hold_balance_item.text if hold_balance_item is not None else 'N/A'
                print(f"Total Balance: {balance}")
                print(f"Hold Balance: {hold_balance}")
            else:
                print(f"Request failed with response code: {response_code_item.text if response_code_item is not None else 'Unknown'}")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)


def check_domain_belongs(domain):
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="object">DOMAIN</item>
            <item key="action">BELONGS_TO_RSP</item>
            <item key="attributes">
             <dt_assoc>
                    <item key="domain">{domain}</item>
             </dt_assoc>
            </item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    #print(xml)

    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            root = ET.fromstring(response_text)
            is_success_item = root.find(".//item[@key='is_success']")
            response_code_item = root.find(".//item[@key='response_code']")
            response_text_item = root.find(".//item[@key='response_text']")
            belongs_to_rsp_item = root.find(".//item[@key='belongs_to_rsp']")
            domain_expdate_item = root.find(".//item[@key='domain_expdate']")

            if is_success_item is not None and is_success_item.text == '1':
                if belongs_to_rsp_item is not None and belongs_to_rsp_item.text == '1':
                    domain_expdate = domain_expdate_item.text if domain_expdate_item is not None else 'N/A'
                    print(f"The domain {domain} belongs to the RSP.")
                    print(f"Expiration Date: {domain_expdate}")
                elif belongs_to_rsp_item is not None and belongs_to_rsp_item.text == '0':
                    print(f"The domain {domain} does not belong to the RSP.")
                else:
                    print(f"Unexpected belongs_to_rsp value: {belongs_to_rsp_item.text if belongs_to_rsp_item is not None else 'None'}")
            else:
                print(f"Request failed with response code: {response_code_item.text if response_code_item is not None else 'Unknown'}")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)

def main():
    parser = argparse.ArgumentParser(description='Check domain availability using OpenSRS API.')
    parser.add_argument('domain', type=str, help='The domain to check availability for')
    args = parser.parse_args()

    check_domain_belongs(args.domain)

if __name__ == '__main__':
    main()
#!/usr/bin/python3

import requests
import hashlib
import argparse
import xml.etree.ElementTree as ET

TEST_MODE = 0

connection_options = {
    'live': {
        # IP whitelisting required
        'reseller_username': 'idragonfly',
        'api_key': '5c71461e543d28f1212d51f505b360c6f879e1f9bbac998f49c4412305d190ae5fcfa0d4bd184b4d1736515f532b0226f207952c3604f54c',
        'api_host_port': 'https://rr-n1-tor.opensrs.net:55443',
    },
    'test': {
        # IP whitelisting not required
        'reseller_username': 'idragonfly',
        'api_key': '5c71461e543d28f1212d51f505b360c6f879e1f9bbac998f49c4412305d190ae5fcfa0d4bd184b4d1736515f532b0226f207952c3604f54c',
        'api_host_port': 'https://horizon.opensrs.net:55443',
    }
}

if TEST_MODE == 1:
    connection_details = connection_options['test']
else:
    connection_details = connection_options['live']

def check_domain_availability(domain):
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="action">LOOKUP</item>
            <item key="object">DOMAIN</item>
            <item key="attributes">
             <dt_assoc>
                    <item key="domain">{domain}</item>
             </dt_assoc>
            </item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    #print(xml)

    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            root = ET.fromstring(response_text)
            status_item = root.find(".//item[@key='status']")
            if status_item is not None and status_item.text == 'taken':
                print(f"The domain {domain} is not available.")
            elif status_item is not None and status_item.text == 'available':
                print(f"The domain {domain} is available.")
            else:
                print(f"Unexpected status: {status_item.text if status_item is not None else 'None'}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)


def send_password(domain_name, send_to, sub_user):
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="object">DOMAIN</item>
            <item key="action">SEND_PASSWORD</item>
            <item key="attributes">
             <dt_assoc>
                    <item key="domain_name">{domain_name}</item>
                    <item key="send_to">{send_to}</item>
                    <item key="sub_user">{sub_user}</item>
             </dt_assoc>
            </item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    #print(xml)

    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            root = ET.fromstring(response_text)
            is_success_item = root.find(".//item[@key='is_success']")
            response_code_item = root.find(".//item[@key='response_code']")
            response_text_item = root.find(".//item[@key='response_text']")

            if is_success_item is not None and is_success_item.text == '1':
                print("Password reset email sent successfully.")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
            else:
                print(f"Failed to send password reset email with response code: {response_code_item.text if response_code_item is not None else 'Unknown'}")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)


def get_price(domain, period, all_periods):
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="object">DOMAIN</item>
            <item key="action">GET_PRICE</item>
            <item key="attributes">
             <dt_assoc>
                    <item key="domain">{domain}</item>
                    <item key="period">{period}</item>
                    <item key="all_periods">{all_periods}</item>
             </dt_assoc>
            </item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    #print(xml)

    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            root = ET.fromstring(response_text)
            is_success_item = root.find(".//item[@key='is_success']")
            response_code_item = root.find(".//item[@key='response_code']")
            response_text_item = root.find(".//item[@key='response_text']")
            price_item = root.find(".//item[@key='price']")
            prices_item = root.find(".//item[@key='prices']")
            is_registry_premium_item = root.find(".//item[@key='is_registry_premium']")
            registry_premium_group_item = root.find(".//item[@key='registry_premium_group']")

            if is_success_item is not None and is_success_item.text == '1':
                price = price_item.text if price_item is not None else 'N/A'
                print(f"Price for {domain} for {period} year(s): {price}")

                if all_periods == '1' and prices_item is not None:
                    prices = prices_item.findall(".//item")
                    for price_info in prices:
                        period = price_info.find(".//item[@key='period']").text if price_info.find(".//item[@key='period']") is not None else 'N/A'
                        price = price_info.find(".//item[@key='price']").text if price_info.find(".//item[@key='price']") is not None else 'N/A'
                        print(f"Price for {domain} for {period} year(s): {price}")

                if is_registry_premium_item is not None:
                    is_registry_premium = is_registry_premium_item.text if is_registry_premium_item is not None else 'N/A'
                    print(f"Is Registry Premium: {is_registry_premium}")

                if registry_premium_group_item is not None:
                    registry_premium_group = registry_premium_group_item.text if registry_premium_group_item is not None else 'N/A'
                    print(f"Registry Premium Group: {registry_premium_group}")

            else:
                print(f"Request failed with response code: {response_code_item.text if response_code_item is not None else 'Unknown'}")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)


def redeem_domain(domain, registrant_ip=None):
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="object">DOMAIN</item>
            <item key="action">REDEEM</item>
            <item key="attributes">
             <dt_assoc>
                    <item key="domain">{domain}</item>
                    {'<item key="registrant_ip">{registrant_ip}</item>' if registrant_ip else ''}
             </dt_assoc>
            </item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    #print(xml)

    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            root = ET.fromstring(response_text)
            is_success_item = root.find(".//item[@key='is_success']")
            response_code_item = root.find(".//item[@key='response_code']")
            response_text_item = root.find(".//item[@key='response_text']")

            if is_success_item is not None and is_success_item.text == '1':
                print("Domain redemption successful.")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
            else:
                print(f"Domain redemption failed with response code: {response_code_item.text if response_code_item is not None else 'Unknown'}")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)


def send_registrant_verification_email(domain):
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="object">DOMAIN</item>
            <item key="action">SEND_REGISTRANT_VERIFICATION_EMAIL</item>
            <item key="attributes">
             <dt_assoc>
                    <item key="domain">{domain}</item>
             </dt_assoc>
            </item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    #print(xml)

    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            root = ET.fromstring(response_text)
            is_success_item = root.find(".//item[@key='is_success']")
            response_code_item = root.find(".//item[@key='response_code']")
            response_text_item = root.find(".//item[@key='response_text']")

            if is_success_item is not None and is_success_item.text == '1':
                print("Registrant verification email sent successfully.")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
            else:
                print(f"Failed to send registrant verification email with response code: {response_code_item.text if response_code_item is not None else 'Unknown'}")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)


def sw_register(domain, reg_type, period, reg_username, reg_password, contact_set, nameserver_list, handle='process', custom_nameservers=0, custom_transfer_nameservers=0, custom_tech_contact=0, f_lock_domain=0, f_parkp='N', f_whois_privacy=0, affiliate_id=None, auto_renew=0, auth_info=None, change_contact=0, comments=None, dns_template=None, encoding_type=None, intended_use=None, link_domains=0, master_order_id=None, owner_confirm_address=None, messaging_language=None, tld_data=None):
    """
    Sends a sw_register request to OpenSRS API to register or transfer a domain.
    
    :param domain: The domain name to be registered or transferred.
    :param reg_type: The type of registration (new, transfer, landrush, sunrise).
    :param period: The registration period in years (1-10).
    :param reg_username: The username of the registrant.
    :param reg_password: The password of the registrant.
    :param contact_set: A dictionary containing contact information for owner, admin, billing, and tech contacts.
    :param nameserver_list: A list of nameservers with their sort order.
    :param handle: Indicates how to process the order (process or save).
    :param custom_nameservers: Use custom nameservers (0 or 1).
    :param custom_transfer_nameservers: Use custom nameservers for transfers (0 or 1).
    :param custom_tech_contact: Use custom tech contact (0 or 1).
    :param f_lock_domain: Lock the domain (0 or 1).
    :param f_parkp: Enable Parked Pages (Y or N).
    :param f_whois_privacy: Enable WHOIS Privacy (0 or 1).
    :param affiliate_id: Affiliate ID for tracking orders.
    :param auto_renew: Set domain to auto-renew (0 or 1).
    :param auth_info: Transfer authcode for the domain.
    :param change_contact: Change contact information during transfer (0 or 1).
    :param comments: Additional notes for the order.
    :param dns_template: DNS template to use.
    :param encoding_type: Encoding type for IDNs.
    :param intended_use: Intended use for .scot registrations.
    :param link_domains: Link domains (0 or 1).
    :param master_order_id: Master order ID for linked domains.
    :param owner_confirm_address: Email address for transfer confirmation.
    :param messaging_language: Messaging language for customer notifications.
    :param tld_data: Additional TLD-specific data.
    """
    # Construct the XML request
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="object">DOMAIN</item>
            <item key="action">SW_REGISTER</item>
            <item key="attributes">
             <dt_assoc>
                    <item key="domain">{domain}</item>
                    <item key="reg_type">{reg_type}</item>
                    <item key="period">{period}</item>
                    <item key="reg_username">{reg_username}</item>
                    <item key="reg_password">{reg_password}</item>
                    <item key="handle">{handle}</item>
                    <item key="custom_nameservers">{custom_nameservers}</item>
                    <item key="custom_transfer_nameservers">{custom_transfer_nameservers}</item>
                    <item key="custom_tech_contact">{custom_tech_contact}</item>
                    <item key="f_lock_domain">{f_lock_domain}</item>
                    <item key="f_parkp">{f_parkp}</item>
                    <item key="f_whois_privacy">{f_whois_privacy}</item>
                    {'<item key="affiliate_id">{affiliate_id}</item>' if affiliate_id else ''}
                    {'<item key="auto_renew">{auto_renew}</item>' if auto_renew else ''}
                    {'<item key="auth_info">{auth_info}</item>' if auth_info else ''}
                    {'<item key="change_contact">{change_contact}</item>' if change_contact else ''}
                    {'<item key="comments">{comments}</item>' if comments else ''}
                    {'<item key="dns_template">{dns_template}</item>' if dns_template else ''}
                    {'<item key="encoding_type">{encoding_type}</item>' if encoding_type else ''}
                    {'<item key="intended_use">{intended_use}</item>' if intended_use else ''}
                    {'<item key="link_domains">{link_domains}</item>' if link_domains else ''}
                    {'<item key="master_order_id">{master_order_id}</item>' if master_order_id else ''}
                    {'<item key="owner_confirm_address">{owner_confirm_address}</item>' if owner_confirm_address else ''}
                    {'<item key="messaging_language">{messaging_language}</item>' if messaging_language else ''}
                    {'<item key="tld_data">{tld_data}</item>' if tld_data else ''}
                    <item key="contact_set">
                        <dt_assoc>
                            {'<item key="owner"><dt_assoc>' + ''.join(f'<item key="{k}">{v}</item>' for k, v in contact_set['owner'].items()) + '</dt_assoc></item>' if 'owner' in contact_set else ''}
                            {'<item key="admin"><dt_assoc>' + ''.join(f'<item key="{k}">{v}</item>' for k, v in contact_set['admin'].items()) + '</dt_assoc></item>' if 'admin' in contact_set else ''}
                            {'<item key="billing"><dt_assoc>' + ''.join(f'<item key="{k}">{v}</item>' for k, v in contact_set['billing'].items()) + '</dt_assoc></item>' if 'billing' in contact_set else ''}
                            {'<item key="tech"><dt_assoc>' + ''.join(f'<item key="{k}">{v}</item>' for k, v in contact_set['tech'].items()) + '</dt_assoc></item>' if 'tech' in contact_set else ''}
                        </dt_assoc>
                    </item>
                    <item key="nameserver_list">
                        <dt_array>
                            {''.join(f'<item key="{i+1}"><dt_assoc><item key="name">{ns["name"]}</item><item key="sortorder">{ns["sortorder"]}</item></dt_assoc></item>' for i, ns in enumerate(nameserver_list))}
                        </dt_array>
                    </item>
             </dt_assoc>
            </item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    # Generate the signature for the request
    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    # Set the headers for the request
    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    # Print the request for debugging purposes
    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    print(xml)

    # Send the POST request to the OpenSRS API
    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    # Print the response from the API
    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            # Parse the XML response
            root = ET.fromstring(response_text)
            is_success_item = root.find(".//item[@key='is_success']")
            response_code_item = root.find(".//item[@key='response_code']")
            response_text_item = root.find(".//item[@key='response_text']")
            admin_email_item = root.find(".//item[@key='admin_email']")
            async_reason_item = root.find(".//item[@key='async_reason']")
            cancelled_orders_item = root.find(".//item[@key='cancelled_orders']")
            error_item = root.find(".//item[@key='error']")
            forced_pending_item = root.find(".//item[@key='forced_pending']")
            id_item = root.find(".//item[@key='id']")
            queue_request_id_item = root.find(".//item[@key='queue_request_id']")
            registration_code_item = root.find(".//item[@key='registration_code']")
            registration_text_item = root.find(".//item[@key='registration_text']")
            transfer_id_item = root.find(".//item[@key='transfer_id']")
            whois_privacy_state_item = root.find(".//item[@key='whois_privacy_state']")

            # Check if the request was successful
            if is_success_item is not None and is_success_item.text == '1':
                print("Domain registration/transfer successful.")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
                if admin_email_item is not None:
                    print(f"Admin Email: {admin_email_item.text}")
                if async_reason_item is not None:
                    print(f"Async Reason: {async_reason_item.text}")
                if cancelled_orders_item is not None:
                    print(f"Cancelled Orders: {cancelled_orders_item.text}")
                if forced_pending_item is not None:
                    print(f"Forced Pending: {forced_pending_item.text}")
                if id_item is not None:
                    print(f"Order ID: {id_item.text}")
                if queue_request_id_item is not None:
                    print(f"Queue Request ID: {queue_request_id_item.text}")
                if registration_code_item is not None:
                    print(f"Registration Code: {registration_code_item.text}")
                if registration_text_item is not None:
                    print(f"Registration Text: {registration_text_item.text}")
                if transfer_id_item is not None:
                    print(f"Transfer ID: {transfer_id_item.text}")
                if whois_privacy_state_item is not None:
                    print(f"WHOIS Privacy State: {whois_privacy_state_item.text}")
            else:
                print(f"Domain registration/transfer failed with response code: {response_code_item.text if response_code_item is not None else 'Unknown'}")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
                if error_item is not None:
                    print(f"Error: {error_item.text}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)


def renew_domain(domain, currentexpirationyear, period, auto_renew, affiliate_id=None, registrant_ip=None, f_parkp=None):
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="object">DOMAIN</item>
            <item key="action">RENEW</item>
            <item key="registrant_ip">{registrant_ip}</item> {'' if registrant_ip else ''}
            <item key="attributes">
             <dt_assoc>
                    <item key="domain">{domain}</item>
                    <item key="currentexpirationyear">{currentexpirationyear}</item>
                    <item key="period">{period}</item>
                    <item key="auto_renew">{auto_renew}</item>
                    {'<item key="affiliate_id">{affiliate_id}</item>' if affiliate_id else ''}
                    {'<item key="f_parkp">{f_parkp}</item>' if f_parkp else ''}
                    <item key="handle">process</item>
             </dt_assoc>
            </item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    #print(xml)

    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            root = ET.fromstring(response_text)
            is_success_item = root.find(".//item[@key='is_success']")
            response_code_item = root.find(".//item[@key='response_code']")
            response_text_item = root.find(".//item[@key='response_text']")
            admin_email_item = root.find(".//item[@key='admin_email']")
            auto_renew_item = root.find(".//item[@key='auto_renew']")
            id_item = root.find(".//item[@key='id']")
            order_id_item = root.find(".//item[@key='order_id']")
            queue_request_id_item = root.find(".//item[@key='queue_request_id']")
            registration_expiration_date_item = root.find(".//item[@key='registration expiration date']")

            if is_success_item is not None and is_success_item.text == '1':
                print("Domain renewal successful.")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
                if admin_email_item is not None:
                    print(f"Admin Email: {admin_email_item.text}")
                if auto_renew_item is not None:
                    print(f"Auto Renew: {'Yes' if auto_renew_item.text == '1' else 'No'}")
                if id_item is not None:
                    print(f"Domain ID: {id_item.text}")
                if order_id_item is not None:
                    print(f"Order ID: {order_id_item.text}")
                if queue_request_id_item is not None:
                    print(f"Queue Request ID: {queue_request_id_item.text}")
                if registration_expiration_date_item is not None:
                    print(f"Registration Expiration Date: {registration_expiration_date_item.text}")
            else:
                print(f"Domain renewal failed with response code: {response_code_item.text if response_code_item is not None else 'Unknown'}")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)


def revoke_domain(domain, notes=None):
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="object">DOMAIN</item>
            <item key="action">REVOKE</item>
            <item key="attributes">
             <dt_assoc>
                    <item key="domain">{domain}</item>
                    {'<item key="notes">{notes}</item>' if notes else ''}
             </dt_assoc>
            </item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    #print(xml)

    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            root = ET.fromstring(response_text)
            is_success_item = root.find(".//item[@key='is_success']")
            response_code_item = root.find(".//item[@key='response_code']")
            response_text_item = root.find(".//item[@key='response_text']")
            charge_item = root.find(".//item[@key='charge']")
            total_refund_item = root.find(".//item[@key='total_refund']")
            deletion_id_item = root.find(".//item[@key='deletion_id']")

            if is_success_item is not None and is_success_item.text == '1':
                print("Domain revocation successful.")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
                if charge_item is not None:
                    print(f"Charge: {'Yes' if charge_item.text == '1' else 'No'}")
                if total_refund_item is not None:
                    print(f"Total Refund: {total_refund_item.text}")
                if deletion_id_item is not None:
                    print(f"Deletion ID: {deletion_id_item.text}")
            else:
                print(f"Domain revocation failed with response code: {response_code_item.text if response_code_item is not None else 'Unknown'}")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)


def set_domain_affiliate_id(domain, affiliate_id):
    xml = f'''
    <?xml version='1.0' encoding='UTF-8' standalone='no' ?>
    <!DOCTYPE OPS_envelope SYSTEM 'ops.dtd'>
    <OPS_envelope>
    <header>
        <version>0.9</version>
    </header>
    <body>
    <data_block>
        <dt_assoc>
            <item key="protocol">XCP</item>
            <item key="object">DOMAIN</item>
            <item key="action">SET_DOMAIN_AFFILIATE_ID</item>
            <item key="attributes">
             <dt_assoc>
                    <item key="domain">{domain}</item>
                    <item key="affiliate_id">{affiliate_id}</item>
             </dt_assoc>
            </item>
        </dt_assoc>
    </data_block>
    </body>
    </OPS_envelope>
    '''

    md5_obj = hashlib.md5()
    md5_obj.update((xml + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    md5_obj = hashlib.md5()
    md5_obj.update((signature + connection_details['api_key']).encode())
    signature = md5_obj.hexdigest()

    headers = {
        'Content-Type': 'text/xml',
        'X-Username': connection_details['reseller_username'],
        'X-Signature': signature,
    }

    print(f"Request to {connection_details['api_host_port']} as reseller {connection_details['reseller_username']}:")
    #print(xml)

    r = requests.post(connection_details['api_host_port'], data=xml, headers=headers)

    print("Response:")
    if r.status_code == requests.codes.ok:
        response_text = r.text
        try:
            root = ET.fromstring(response_text)
            is_success_item = root.find(".//item[@key='is_success']")
            response_code_item = root.find(".//item[@key='response_code']")
            response_text_item = root.find(".//item[@key='response_text']")

            if is_success_item is not None and is_success_item.text == '1':
                print("Affiliate ID assigned successfully.")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
            else:
                print(f"Failed to assign affiliate ID with response code: {response_code_item.text if response_code_item is not None else 'Unknown'}")
                print(f"Response text: {response_text_item.text if response_text_item is not None else 'No response text'}")
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            print(response_text)
    else:
        print(r.status_code)
        print(r.text)

