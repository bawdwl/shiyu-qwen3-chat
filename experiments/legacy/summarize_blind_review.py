"""由组织者使用：只汇总实际填写的匿名评审，不产生评分。"""
import argparse
import json
from pathlib import Path
from prepare_blind_review import summarize_reviews


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', required=True)
    parser.add_argument('--reviews', required=True)
    args = parser.parse_args()
    key = json.loads(Path(args.key).read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in Path(args.reviews).read_text(encoding='utf-8').splitlines() if line.strip()]
    print(json.dumps(summarize_reviews(rows, key['mapping'], key['batch_id']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
