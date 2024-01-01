import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def mean(values):
    mean = 0
    for value in values:
        mean += value
    mean /= len(values)
    return mean


def median(values):
    values.sort()
    n = len(values)
    if n % 2 == 1:
        return values[n // 2 + 1]
    else:
        return (values[n // 2] + values[n // 2 + 1]) / 2


def sample_range(values):
    n = len(values)
    return values[n - 1] - values[0]


def variance(values):
    n = len(values)
    m = mean(values)
    variance = 0
    for value in values:
        variance += (value - m) ** 2
    variance /= (n - 1)
    return variance


def average_deviation(values):
    n = len(values)
    deviation = 0
    m = mean(values)
    for value in values:
        deviation += abs(value - m)
    return deviation / n

def left_hinge(values):
    med = median(values)
    lower_values = []
    for value in values:
        if value <= med:
            lower_values.append(value)
    return median(lower_values)


def right_hinge(values):
    med = median(values)
    upper_values = []
    for value in values:
        if value > med:
            upper_values.append(value)
    return median(upper_values)


# Connect to SQLite database


# # histogram
# plt.hist(x_values, bins=15)
# plt.title("Histogram")
# plt.xlabel("Laps driven")
# plt.ylabel("Amount")
# plt.show()
#
# # boxplot
# plt.boxplot(x_values)
# plt.title("Boxplot")
# plt.xlabel("Laps driven")
# plt.show()
#
# # vioplot
# sns.violinplot(x_values)
# plt.title("Violinplot")
# plt.show()


# linear regression

def linear_regression(x_values, y_values):
    B = [x for x in range(1, 15000, 1000)]
    best_model_b = B[0]
    best_model_a = 0
    best_error = float("inf")
    learning_rate = 1e-7
    epochs = 2000

    n = x_values.size
    for b in B:
        a = best_model_a
        for _ in range(epochs):
            dl_da = 0
            dl_db = 0
            for i in range(n):
                dl_da += (a * x_values[i] + b - y_values[i]) * x_values[i]
                dl_db += (a * x_values[i] + b - y_values[i])
            # gradient descent
            a -= 2/n * dl_da * learning_rate
            b -= 2/n * dl_db * learning_rate

        print(a, b)
        error = 0
        for i in range(x_values.size):
            error += (a * x_values[i] + b - y_values[i]) ** 2
        if error < best_error:
            best_error = error
            best_model_b = b
            best_model_a = a
    print(best_model_a, best_model_b)

    plt.scatter(x_values, y_values)
    plt.plot(list(range(1993, 2023)), [best_model_a * x + best_model_b for x in range(1993, 2023)], color="red")
    plt.show()


# linear_regression(x_values, y_values)

db_name = 'f1.db'
connection = sqlite3.connect(db_name)

# Query the data from the table
query = "SELECT ra.name, ra.year, d.forename || ' ' || d.surname AS driver_name, c.name, re.positionOrder, re.laps, s.status FROM results re INNER JOIN races ra ON ra.raceId = re.raceId INNER JOIN drivers d ON d.driverId = re.driverId INNER JOIN constructors c ON c.constructorId = re.constructorId INNER JOIN status s ON s.statusId = re.statusId"
result = connection.execute(query).fetchall()

# Get column names from the cursor description
columns = ["race_name", "race_year", "driver_name", "constructor_name", "driver_final_position", "laps_driven", "ending"]

# Create a DataFrame
df = pd.DataFrame(result, columns=columns)

# Close the connection
connection.close()

# Print the DataFrame
print(df)

dnfs2 = df[~df["ending"].str.contains(r'\b(?:Lap|Laps|Finished)\b', case=False, regex=True)]['race_year']
grouped_dnfs = dnfs2.value_counts().reset_index(name='count')

linear_regression(grouped_dnfs['race_year'], grouped_dnfs['count'])