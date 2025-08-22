import requests

def main():
    r = requests.get("http://localhost:8000/health")
    if r.status_code != 200:
        print(r.status_code)
    print(r.json())

if __name__ == "__main__":
    main()