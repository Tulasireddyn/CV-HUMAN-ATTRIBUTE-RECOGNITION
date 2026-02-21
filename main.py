
import argparse

def main():
    parser = argparse.ArgumentParser(description="Human Attribute Recognition")
    parser.add_argument('--task', type=str, required=True, help='Task: age, gender, emotion, clothing, pose')
    args = parser.parse_args()
    print(f"Running task: {args.task}")

if __name__ == "__main__":
    main()
