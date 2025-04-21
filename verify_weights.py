import json
import argparse


def verify_weights(weight_file):
    with open(weight_file, 'r') as f:
        multi_horizon_portfolio = json.load(f)
    abs_weight = 0
    sum_weight = 0
    neg_weight = 0
    pos_weight = 0
    for stock in multi_horizon_portfolio:
        #print(f"Stock: {stock}")
        weight = stock['weight']
        if weight < 0:
            neg_weight += weight
        else:
            pos_weight += weight
        abs_weight += abs(weight)
        sum_weight += weight

    print(f"Sum of weights: {sum_weight}")
    print(f"Sum of absolute weights: {abs_weight}")
    print(f"Sum of negative weights: {neg_weight}")
    print(f"Sum of positive weights: {pos_weight}")

def merge_weights(weight_file):
    with open(weight_file, 'r') as f:
        multi_horizon_portfolio = json.load(f)
    merged_weights = {}
    for stock in multi_horizon_portfolio:
        stock_name = stock['id']
        weight = stock['weight']
        if '_' in stock_name:
            stock_name = stock_name.split('_')[0]

        if stock_name in merged_weights:
            merged_weights[stock_name] += weight
        else:
            merged_weights[stock_name] = weight
    # sort the merged weights through the absolute value of weights
    merged_weights = dict(sorted(merged_weights.items(), key=lambda item: abs(item[1]), reverse=True))
    return merged_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify weights from a weight file.')
    parser.add_argument('filename', type=str, help='Path to the weight JSON file')

    args = parser.parse_args()

    verify_weights(args.filename)
    merged_weights = merge_weights(args.filename)
    print("Merged weights:")
    for stock, weight in merged_weights.items():
        print(f"{stock}: {weight}")
    print("Verification complete.")
