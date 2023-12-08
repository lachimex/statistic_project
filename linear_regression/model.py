import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
file_path = 'Salary_dataset.csv'
data = pd.read_csv(file_path)

# Extract columns for the plot
x_values = data['YearsExperience']
y_values = data['Salary']

# initial values
a = 1
b = 1
learning_rate = 0.01


epochs = 300

for _ in range(epochs):
    dl_da = 0
    dl_db = 0
    for i in range(x_values.size):
        dl_da += 2 * (a * x_values[i] + b - y_values[i]) * x_values[i]
        dl_db += 2 * (a * x_values[i] + b - y_values[i])
    dl_da /= x_values.size
    dl_db /= x_values.size
    #gradient descent
    a -= dl_da * learning_rate
    b -= dl_db * learning_rate

plt.scatter(x_values, y_values)
plt.plot(list(range(1, 12)), [a * x + b for x in range(1, 12)], color="red")
plt.title('Salary vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()