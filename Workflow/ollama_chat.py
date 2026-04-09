"""
Simple chat script for Ollama server.
Connects to Ollama at 172.24.77.77 and uses the gemma4:e4b model.
"""

import requests
import json
import sys

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    
    from prompt_toolkit import prompt
    from prompt_toolkit.layout.processors import Processor, Transformation

    class ArabicReshaperProcessor(Processor):
        def apply_transformation(self, ti):
            text = ti.document.text
            if text:
                reshaped = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped)
                return Transformation([('', bidi_text)])
            return Transformation(ti.fragments)
            
    HAS_BIDI = True
except ImportError:
    HAS_BIDI = False

OLLAMA_URL = "http://172.24.77.77:11434"
MODEL = "gemma4:e4b"


def check_connection():
    """Verify the Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  Available models: {', '.join(models) if models else 'none found'}")
        return True
    except Exception as e:
        print(f"❌ Cannot reach Ollama at {OLLAMA_URL}: {e}")
        return False


def chat(messages: list[dict]) -> str:
    """Send a chat request to Ollama (non-streaming) and return the response."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        reply = data.get("message", {}).get("content", "")
        
        # Reshape Arabic text if libraries are available
        if HAS_BIDI:
            reshaped_text = arabic_reshaper.reshape(reply)
            bidi_text = get_display(reshaped_text)
            print(bidi_text)
        else:
            print(reply)
            
        return reply

    except requests.ConnectionError as e:
        print(f"\n❌ Connection error: {e}")
        return ""
    except requests.Timeout:
        print("\n❌ Request timed out (300s). The model may be loading.")
        return ""
    except requests.HTTPError as e:
        print(f"\n❌ HTTP error: {e}")
        try:
            print(f"   Detail: {response.text}")
        except Exception:
            pass
        return ""
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return ""


def main():
    print("=" * 50)
    print(f"  Ollama Chat  —  Model: {MODEL}")
    print(f"  Server: {OLLAMA_URL}")
    print("  Type 'exit' or 'quit' to end the chat.")
    print("=" * 50)
    print()

    print("Checking connection...")
    if not check_connection():
        sys.exit(1)
    print()

    messages: list[dict] = []

    while True:
        try:
            if HAS_BIDI:
                user_input = prompt("> ", input_processors=[ArabicReshaperProcessor()]).strip()
            else:
                user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break

        messages.append({"role": "user", "content": user_input})

        assistant_reply = chat(messages)

        if assistant_reply:
            messages.append({"role": "assistant", "content": assistant_reply})
        else:
            # Remove the failed user message so history stays clean
            messages.pop()

        print()


if __name__ == "__main__":
    main()
