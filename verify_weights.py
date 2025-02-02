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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify weights from a weight file.')
    parser.add_argument('filename', type=str, help='Path to the weight JSON file')
    
    args = parser.parse_args()
    
    verify_weights(args.filename)
