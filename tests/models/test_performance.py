from my_krml_25552249.models.performance import plot_confusion_matrix

def test_plot_runs_without_error():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    plot_confusion_matrix(y_true, y_pred, labels=[0, 1])
