import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

tf.random.set_seed(42)
np.random.seed(42)
x1 = np.random.rand(4000)*10
x2 = np.random.rand(4000)*10
x3 = np.random.rand(4000)*10

x1_int = x1.astype(int) - 5
x2_int = x2.astype(int) - 5
x3_int = x3.astype(int) - 5

x1_str = x1_int.astype(str)
x2_str = x2_int.astype(str)
x3_str = x3_int.astype(str)

unique_expressions = set()

for i in range(len(x1)):
  n = np.random.rand(1)
  if n < 0.25:
    opp1 = " + "
    opp2 = " + "
  elif n > 0.25 and n < 0.5:
    opp1 = " + "
    opp2 = " - "
  elif n > 0.75:
    opp1 = " - "
    opp2 = " + "
  else:
    opp1 = " - "
    opp2 = " - "
  unique_expressions.add(x1_str[i] + opp1 + x2_str[i] + opp2 + x3_str[i])

x = list(unique_expressions) # Convert the set back to a list
print(x[0])
print(len(x))
y = []

for expression in x:
  result = float(eval(expression))
  y.append(result)

print(y[0])
def tokenizer(input_list): # Changed parameter name to avoid confusion with global x
  #tokenizer by hand
  #tokens = (len(input_list), 5)
  # Create a copy of the input list to avoid modifying the original
  tokenized_x = [expression.split(" ") for expression in input_list]


  for i in range(len(tokenized_x)):
    for j in range(len(tokenized_x[i])):
      if j % 2 == 0:  # Check if the index is even
        tokenized_x[i][j] = np.float32(tokenized_x[i][j])
      else:  # The index is uneven, it's an operator
        if tokenized_x[i][j] == "+":
          tokenized_x[i][j] = np.float32(1)
        else:
          tokenized_x[i][j] = np.float32(0)
    padding_count = 15 - len(tokenized_x[i])
    for _ in range(padding_count): # Use a throwaway variable
      tokenized_x[i].append(np.float32(0.5))
  tokenized_x = np.array(tokenized_x)
  return tokenized_x
x = tokenizer(x)
# Generate all possible expressions
all_possible_expressions = set()
for num1 in range(-5, 5): # Range -5 to 4
    for num2 in range(-5, 5): # Range -5 to 4
        for num3 in range(-5, 5): # Range -5 to 4
            for op1 in [" + ", " - "]:
                for op2 in [" + ", " - "]:
                    expression = str(num1) + op1 + str(num2) + op2 + str(num3)
                    all_possible_expressions.add(expression)

# Find expressions not in x
expressions_not_in_x = all_possible_expressions - unique_expressions
expressions_not_in_x = list(expressions_not_in_x)
print("\nExpressions not in x:")
print(expressions_not_in_x[0]) # Convert back to a list for printing
if len(expressions_not_in_x)+len(x) == 4000: print(True)
print(len(expressions_not_in_x))

x_test = np.array(tokenizer(expressions_not_in_x))

y_test = []
for expression in expressions_not_in_x:
    y_test.append(float(eval(expression)))

print(y_test[0])
#a list of expressions outside the number range
outsideExpr = set()
range1= range(5, 9)
range2= range(-8, -4)
comboRange = list(range1) + list(range2)
for num1 in comboRange:
    for num2 in comboRange:
        for num3 in comboRange:
            for op1 in [" + ", " - "]:
                for op2 in [" + ", " - "]:
                    expression = str(num1) + op1 + str(num2) + op2 + str(num3)
                    outsideExpr.add(expression)

outsideExpr = list(outsideExpr)
print(len(tokenizer(outsideExpr)[0]))

out_x_test = tokenizer(outsideExpr)
out_y_test = []
for i in outsideExpr:
  result = float(eval(i))
  out_y_test.append(result)
minNums = 2
maxNums = 8
amountNums = 100
num_terms = np.arange(minNums-1,maxNums)
longer_Exps = []
for j in num_terms:
  for p in range(amountNums):
    longer_Exp = ""
    for i in range(j):
      longer_Exp += str(np.random.randint(-5,6))
      longer_Exp += np.random.choice([" + ", " - "])
    longer_Exp += str(np.random.randint(-5,6))
    longer_Exps.append(longer_Exp)

longer_Exps = list(longer_Exps)
long_x_test = tokenizer(longer_Exps)
long_y_test = []
for i in longer_Exps:
  result = float(eval(i))
  long_y_test.append(result)
print(long_y_test[0])
print(long_x_test[0])
x_train, x_val, y_train, y_val = \
    train_test_split(x, y, train_size=0.75)
x_train, x_val, y_train, y_val = \
    np.array(x_train), np.array(x_val), np.array(y_train), np.array(y_val)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    monitor='mse',
    mode = "min"
)