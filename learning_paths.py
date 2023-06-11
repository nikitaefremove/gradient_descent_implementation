import matplotlib.pyplot as plt

# Create figure
fig = plt.figure()

# Change a size of figure
fig.set_size_inches(13, 10)

# Create arrays with different thresholds and learning_rates
thresholds = [1e-2, 1e-3, 1e-4, 1e-5]
learning_rates = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

# Create a loop over all thresholds
for i, threshold in enumerate(thresholds):
    # Create an array with MSE values
    Q_values = []

    # Create subplots to the figure
    ax_ = fig.add_subplot(2, 2, i + 1)

    # Create a loop over all learning_rates
    for rate in learning_rates:
        # Create a model of GradientDescentMse class
        model = GradientDescentMse(samples=X, targets=Y, learning_rate=rate, threshold=threshold)
        model.add_constant_feature()
        model.learn()

        # Create a learning_path variable as a dictionary from iteration_loss_dict
        learning_path = model.iteration_loss_dict

        # Plot graphs with number of iteration as X-axis and MSE as Y-axis
        plt.plot(learning_path.keys(), learning_path.values())
        plt.ylim(0, 100)
        plt.xlim(0, 2000)
        # Append MSE values to Q_values
        Q_values.append(str(round(list(learning_path.values())[-1], ndigits=4)))

    # Adding legends, titles, and labels to the graph
    plt.ylabel('Mean squared error')
    plt.xlabel('Number of iteration')
    plt.legend([f'Learning rate equals to {learning_rates[i]}' + ' with Q = ' + str(Q_values[i]) for i in
                range(len(learning_rates))])
    ax_.set_title(f'Threshold = {threshold}')
    ax_.annotate(f'Q = {Q_values}', xy=(0.3, 0.6), xycoords='axes fraction')

# Automatically adjust the positions of subplots
fig.tight_layout()

plt.show()
