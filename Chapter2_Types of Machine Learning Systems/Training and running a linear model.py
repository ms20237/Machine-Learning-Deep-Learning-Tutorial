import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',
encoding='latin1', na_values="n/a")


def prepare_country_stats(oecd_bli, gdp_per_capita):
    # Filter the OECD data to select the relevant indicators
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    # Merge the datasets
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    # Filter out non-OECD countries
    full_country_stats = full_country_stats[["GDP per capita", 'Life satisfaction']]
    return full_country_stats


# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]


# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Select a linear model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)
model.intercept_, model.coef_

# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]


# Predict method for set results
y_Pred = model.predict(x_test)

# Visualising the Training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, model.predict(x_train), color='blue')
plt.title('Linear Regression method')
plt.xlabel('GDP per capita')
plt.ylabel('Life satisfaction')
plt.show()



# Select a linear model
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(x_train, y_train)

# Predict method for set results
y_Pred = model.predict(x_test)


# Visualising the Training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, model.predict(x_train), color='blue')
plt.title('k-Nearest Neighbors method')
plt.xlabel('GDP per capita')
plt.ylabel('Life satisfaction')
plt.show()


# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus' GDP per capita
print(model.predict(X_new))
