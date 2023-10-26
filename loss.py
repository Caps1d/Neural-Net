import numpy as np


class Loss:
    def calculate(self, output, y):
        # Loss setup
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        # Accuracy setup
        preds = np.argmax(output, axis=1)
        bools = [x == y for x, y in zip(preds, y)]
        accuracy = np.mean(bools)
        print(accuracy)

        return data_loss


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred: list, y_true: list):
        # The first line is only needed for testing
        # in practice all inpput will be of np array type
        y_true = np.asarray(y_true)
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        neg_log_conf = -np.log(correct_confidences)
        return neg_log_conf
