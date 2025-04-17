import pandas as pd
import seaborn as sns
from itertools import combinations
from collections import defaultdict

# Load inbuilt dataset
df = sns.load_dataset('titanic')
# We'll treat 'passenger class', 'sex', and 'embarked' as items "in a basket" (like items bought together)
df = df[['pclass', 'sex', 'embarked']].dropna()

# Convert each row to a transaction (basket of categorical attributes)
transactions = df.astype(str).values.tolist()
# Step 1: Create itemsets and count frequency
def get_frequent_itemsets(transactions, min_support=0.1):
    item_counts = defaultdict(int)
    transaction_list = list(map(set, transactions))
    total_tx = len(transaction_list)
    freq_itemsets = []

    # Generate all combinations of size 1 to 3
    for size in range(1, 4):
        for transaction in transaction_list:
            for itemset in combinations(transaction, size):
                item_counts[itemset] += 1

    # Filter based on min support
    for itemset, count in item_counts.items():
        support = count / total_tx
        if support >= min_support:
            freq_itemsets.append((itemset, support))

    return freq_itemsets
# Step 2: Generate association rules from frequent itemsets
def generate_rules(freq_itemsets, min_confidence=0.6):
    rules = []
    itemset_dict = {frozenset(k): v for k, v in freq_itemsets}

    for itemset in itemset_dict:
        if len(itemset) >= 2:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if consequent:
                        confidence = itemset_dict[itemset] / itemset_dict.get(antecedent, 1)
                        if confidence >= min_confidence:
                            rules.append((set(antecedent), set(consequent), confidence))

    return rules
# Run Apriori
frequent_itemsets = get_frequent_itemsets(transactions, min_support=0.1)
rules = generate_rules(frequent_itemsets, min_confidence=0.6)
# Display results
print("Frequent Itemsets:")
for item, support in frequent_itemsets:
    print(f"{set(item)}: support = {support:.2f}")
print("\nAssociation Rules:")
for antecedent, consequent, confidence in rules:
    print(f"{antecedent} => {consequent} (confidence = {confidence:.2f})")