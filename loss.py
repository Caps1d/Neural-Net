import numpy as np


class Loss:
    def calculate(self, output, y):
        # Loss setup
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        # Accuracy setup
        preds = np.argmax(output, axis=1)
        # The reason for using ternary here is because hypothetically we might be dealing with
        # a list of labels rather than a np array with one-hot encoded values
        class_targets = y if len(np.asarray(y).shape) == 1 else np.argmax(y, axis=1)
        accuracy = np.mean(preds == class_targets)
        print(f"Accuracy: {accuracy}")

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
