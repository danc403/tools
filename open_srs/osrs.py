#!/usr/bin/env python3
""" Available commands:
balance, check-domain, check-avail, send-password, get-price, redeem, send-verification, register, renew, revoke, set-affiliate
"""

import argparse
import json
import os
import sys

from utils import parse_arguments, TEST_MODE, connection_details
from domain_management import (
    get_balance,
    check_domain_belongs,
    check_domain_availability,
    send_password,
    get_price,
    redeem_domain,
    send_registrant_verification_email,
    sw_register,
    renew_domain,
    revoke_domain,
    set_domain_affiliate_id
)

def main():
    args = parse_arguments()
    
    if args.command == 'balance':
        get_balance()
    elif args.command == 'check-domain':
        check_domain_belongs(args.domain)
    elif args.command == 'check-avail':
        check_domain_availability(args.domain)
    elif args.command == 'send-password':
        send_password(args.domain_name, args.send_to, args.sub_user)
    elif args.command == 'get-price':
        all_value = 1 if args.all_periods else 0
        get_price(args.domain, args.period, str(all_value))
    elif args.command == 'redeem':
        redeem_domain(args.domain, args.registrant_ip)
    elif args.command == 'send-verification':
        send_registrant_verification_email(args.domain)
    elif args.command == 'register':
        contact_set = json.loads(args.contact_set)
        nameserver_list = json.loads(args.nameserver_list)
        sw_register(args.domain, args.reg_type, args.period, args.reg_username, args.reg_password, contact_set, nameserver_list, handle=args.handle, custom_nameservers=args.custom_nameservers, custom_transfer_nameservers=args.custom_transfer_nameservers, custom_tech_contact=args.custom_tech_contact, f_lock_domain=args.f_lock_domain, f_parkp=args.f_parkp, f_whois_privacy=args.f_whois_privacy, affiliate_id=args.affiliate_id, auto_renew=args.auto_renew, auth_info=args.auth_info, change_contact=args.change_contact, comments=args.comments, dns_template=args.dns_template, encoding_type=args.encoding_type, intended_use=args.intended_use, link_domains=args.link_domains, master_order_id=args.master_order_id, owner_confirm_address=args.owner_confirm_address, messaging_language=args.messaging_language, tld_data=args.tld_data)
    elif args.command == 'renew':
        renew_domain(args.domain, currentexpirationyear=args.currentexpirationyear, period=args.period, auto_renew=args.auto_renew, affiliate_id=args.affiliate_id, registrant_ip=args.registrant_ip, f_parkp=args.f_parkp)
    elif args.command == 'revoke':
        revoke_domain(args.domain, notes=args.notes)
    elif args.command == 'set-affiliate':
        set_domain_affiliate_id(args.domain, args.affiliate_id)
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()
