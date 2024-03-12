# Some plot functions
from .packages import *

# Figure parameters
figsize = ((20, 10), (10, 10))
dpi = 150


# Plot the prediction curve and prediction scatter
def curve_scatter(y_test, y_pred, title='Title', figsize=figsize, dpi=dpi):
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Compute the performance index
    r2 = [round(100 * _, 3) for _ in r2_score(y_test, y_pred, multioutput='raw_values')]
    rmse = [round(np.sqrt(_), 3) for _ in mean_squared_error(y_test, y_pred, multioutput='raw_values')]
    mae = [round(_, 3) for _ in mean_absolute_error(y_test, y_pred, multioutput='raw_values')]

    for i in range(y_test.shape[1]):
        # Plot the prediction curve
        plt.figure(figsize=figsize[0], dpi=dpi)
        
        plt.plot(y_test[:, i], label='Ground Truth')
        plt.plot(y_pred[:, i], label='Prediction')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title(title + ' (QV-{}) (R2: {:.3f}%, RMSE: {:.3f}, MAE: {:.3f})'.format(i + 1, r2[i], rmse[i], mae[i]))
        plt.grid()
        plt.legend()

        # Plot the prediction scatter
        plt.figure(figsize=figsize[1], dpi=dpi)
        plt.scatter(y_test[:, i], y_pred[:, i], label='Samples')
        plt.plot(y_test[:, i], y_test[:, i], 'r', label='Isoline')
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.title(title + ' (QV-{}) (R2: {:.3f}%, RMSE: {:.3f}, MAE: {:.3f})'.format(i + 1, r2[i], rmse[i], mae[i]))
        plt.grid()
        plt.legend()
    plt.show()

    return r2, rmse, mae


# Plot confusion matrix
def confusion(y_test, y_pred, title='Title', figsize=figsize[1], dpi=dpi):
    # Compute the performance index
    acc = round(100 * accuracy_score(y_test, y_pred), 2)

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(cm, cmap='YlGnBu', annot=True, fmt='d')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title(title + ' (Accuracy: {:.2f}%)'.format(acc))
    plt.show()

    return acc


# Plot scatter
def scatter(pc, label, title='Title', figsize=figsize[1], dpi=dpi):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(pc[:, 0], pc[:, 1], c=label, cmap='tab10')
    plt.xlabel('Component_1')
    plt.ylabel('Component_2')
    plt.title(title)
    plt.show()
