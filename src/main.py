import argparse
import asyncio
import sys
from src.api.main import app

def main():
    parser = argparse.ArgumentParser(description="Loan Originator CLI")
    parser.add_argument("--worker", action="store_true", help="Start the background worker for pipeline processing")
    parser.add_argument("--api", action="store_true", help="Start the FastAPI service")
    parser.add_argument("--status", type=str, help="Get status of an application")

    args = parser.parse_args()

    if args.worker:
        print("Starting Loan Originator background worker (RabbitMQ/SQS)...")
        # Logic for starting consumer
    elif args.api:
        print("Starting Loan Originator API...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.status:
        # TODO: Lookup application check
        print(f"Status for {args.status}: PENDING")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
