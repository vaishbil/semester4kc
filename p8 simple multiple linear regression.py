import seaborn as sns 
import statsmodels.api as sm 
# Load dataset 
tips = sns.load_dataset('tips') 
# Simple Linear Regression (Tip ~ Total Bill) 
X = tips[['total_bill']] 
y = tips['tip'] 
X = sm.add_constant(X) 
model_simple = sm.OLS(y, X).fit() 
print(model_simple.summary()) 
# Multiple Linear Regression (Tip ~ Total Bill + Size) 
X_multi = tips[['total_bill', 'size']] 
X_multi = sm.add_constant(X_multi) 
model_multi = sm.OLS(y, X_multi).fit() 
print(model_multi.summary())