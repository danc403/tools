import requests
import json
import argparse

def generate_text(host, port, endpoint, prompt=None, prompts=None, max_new_tokens=50, temperature=0.7, top_p=0.95, do_sample=True):
    url = f"http://{host}:{port}/{endpoint}"  # Construct URL dynamically

    headers = {'Content-Type': 'application/json'}
    data = {
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'do_sample': do_sample
    }

    if prompt:
        data['prompt'] = prompt
    elif prompts:
        data['prompts'] = prompts

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()

        if prompt:
            if 'generated_text' in result:
                return result['generated_text']
            elif 'error' in result:
                print(f"Error from server: {result['error']}")
                return None
            else:
                print("Unexpected response from server.")
                return None
        elif prompts:
            if 'generated_texts' in result:
                return result['generated_texts']
            elif 'error' in result:
                print(f"Error from server: {result['error']}")
                return None
            else:
                print("Unexpected response from server.")
                return None


    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response from server: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for Qwen2.5-0.5B-Instruct-Q8_0 API")
    parser.add_argument("-H", "--host", default="127.0.0.1", help="Host of the API server")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port of the API server")
    parser.add_argument("-e", "--endpoint", default="generate", help="API endpoint (e.g., generate, info)")  # Default endpoint
    parser.add_argument("-pr","--prompt", help="Input prompt")
    parser.add_argument("-ps", "--prompts", nargs="+", help="List of input prompts (for batch generation)")
    parser.add_argument("-n", "--max_new_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("-tp", "--top_p", type=float, default=0.95, help="Top-p (nucleus sampling) value")
    parser.add_argument("-s", "--do_sample", action="store_true", default=True, help="Use sampling")

    args = parser.parse_args()

    if args.endpoint == "info":  # Handle info/help endpoint
        url = f"http://{args.host}:{args.port}" # info is at the root url
        try:
            response = requests.get(url)  # Use GET for info
            response.raise_for_status()
            print(json.dumps(response.json(), indent=4)) # Print formatted json
        except requests.exceptions.RequestException as e:
            print(f"Error getting info: {e}")
    elif args.prompt or args.prompts: # only try to generate if prompts are provided
        result = generate_text(args.host, args.port, args.endpoint, args.prompt, args.prompts, args.max_new_tokens, args.temperature, args.top_p, args.do_sample)

        if result:
            if isinstance(result, str):
                print(result)
            elif isinstance(result, list):
                for text in result:
                    print(text)
                    print("---")
    else:
        print("Either --prompt or --prompts must be specified for the /generate endpoint.")
