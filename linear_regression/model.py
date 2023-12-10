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


# Load dataset
file_path = 'Salary_dataset.csv'
data = pd.read_csv(file_path)

# Extract columns for the plot
x_values = data['YearsExperience']
y_values = data['Salary']

print("Mean:\n", round(mean(x_values), 2))

print("Median:\n", round(median(x_values.tolist()), 2))

print("Range:\n:", round(sample_range(x_values), 2))

print("Variance:\n", round(variance(x_values), 2))

print("Standard deviation:\n", round(variance(x_values) ** 1 / 2, 2))

print("Average deviation:\n", round(average_deviation(x_values), 2))

print("Right hinge:\n", round(right_hinge(x_values.tolist()),2))

print("Left hinge:\n", round(left_hinge(x_values.tolist()),2))

# histogram
plt.hist(x_values, bins=15)
plt.title("Histogram")
plt.xlabel("Years Experience")
plt.ylabel("Amount")
plt.show()

# boxplot
plt.boxplot(x_values)
plt.title("Boxplot")
plt.xlabel("Years Experience")
plt.show()

# vioplot
sns.violinplot(x_values)
plt.title("Violinplot")
plt.show()


# linear regression

def linear_regression(x_values, y_values):
    a = 1
    b = 1
    learning_rate = 0.01

    epochs = 1000

    for _ in range(epochs):
        dl_da = 0
        dl_db = 0
        for i in range(x_values.size):
            dl_da += 2 / x_values.size * (a * x_values[i] + b - y_values[i]) * x_values[i]
            dl_db += 2 / x_values.size * (a * x_values[i] + b - y_values[i])
        # gradient descent
        a -= dl_da * learning_rate
        b -= dl_db * learning_rate

    print(a, b)
    error = 0
    for i in range(x_values.size):
        error += (a * x_values[i] + b - y_values[i]) ** 2
    print(error)

    plt.scatter(x_values, y_values)
    plt.plot(list(range(1, 12)), [a * x + b for x in range(1, 12)], color="red")
    plt.title('Salary vs Years of Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()


# linear_regression(x_values, y_values)
