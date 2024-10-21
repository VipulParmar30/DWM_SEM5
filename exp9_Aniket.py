# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample data: creating a simple dataset of transactions
# Each row represents a transaction, and each column represents an item.
# A value of 1 indicates that the item is present in that transaction.
data = {
    'Milk': [1, 0, 1, 1, 0],
    'Bread': [1, 1, 0, 1, 1],
    'Eggs': [1, 1, 1, 0, 1],
    'Butter': [0, 1, 0, 1, 1],
    'Cheese': [0, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# Generate frequent itemsets using the Apriori algorithm
# Set min_support to the minimum support value (e.g., 0.6 means the item should appear in 60% of the transactions)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Generate association rules from the frequent itemsets
# Set the metric to 'lift', 'confidence', or 'support' and the corresponding threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the frequent itemsets and the generated rules
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)