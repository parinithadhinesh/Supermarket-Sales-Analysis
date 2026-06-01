from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
X = df[['Quantity', 'Tax 5%', 'Rating']]

y = df['Total']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)
r2 = r2_score(y_test, predictions)

print("R2 Score:", r2)

comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
})

print(comparison.head())
plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()




## Sales Prediction Conclusion

A Linear Regression model was implemented to predict supermarket sales using Quantity, Tax, and Customer Ratings.

The model achieved good prediction performance and demonstrated how machine learning can help businesses forecast revenue and analyze sales behavior.

This project combines:

* Data Cleaning
* Exploratory Data Analysis
* Visualization
* Machine Learning

to provide meaningful business insights from retail sales data.








