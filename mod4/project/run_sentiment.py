import random

import embeddings

import minitorch
from datasets import load_dataset

BACKEND = minitorch.TensorBackend(minitorch.FastOps)


def RParam(*shape):
    """Create a random parameter tensor.

    Args:
        *shape: Variable length argument list of tensor dimensions

    Returns:
        Parameter: A Parameter object containing random values between -0.05 and 0.05
    """
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    """A linear transformation layer.

    Args:
        in_size: Number of input features
        out_size: Number of output features
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        """Forward pass of linear layer.

        Args:
            x: Input tensor of shape (batch_size, in_size)

        Returns:
            Tensor: Output tensor of shape (batch_size, out_size)
        """
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv1d(minitorch.Module):
    """1D convolutional layer.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_width: Width of the convolutional kernel
    """
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input):
        """Forward pass of conv1d layer.

        Args:
            input: Input tensor of shape (batch, in_channels, seq_len)

        Returns:
            Tensor: Output tensor of shape (batch, out_channels, seq_len - kernel_width + 1)
        """
        out = minitorch.conv1d(input, self.weights.value) + self.bias.value
        # out: [batch, out_channels, (seq_len - kernel_width + 1)]
        return out


class CNNSentimentKim(minitorch.Module):
    """CNN for Sentiment classification based on Y. Kim 2014.

    This model implements:
    1. 1D convolution with input_channels=embedding_dim, feature_map_size=100 output channels
       and [3,4,5]-sized kernels followed by ReLU activation
    2. Max-over-time pooling across each feature map
    3. Linear layer to size C (number of classes) with ReLU and 25% Dropout
    4. Sigmoid over class dimension

    Args:
        feature_map_size: Number of feature maps (default: 100)
        embedding_size: Size of word embeddings (default: 50)
        filter_sizes: List of kernel sizes (default: [3,4,5])
        dropout: Dropout rate (default: 0.25)
    """

    def __init__(
        self,
        feature_map_size=100,
        embedding_size=50,
        filter_sizes=[3, 4, 5],
        dropout=0.25,
    ):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.conv1 = Conv1d(embedding_size, feature_map_size, filter_sizes[0])
        self.conv2 = Conv1d(embedding_size, feature_map_size, filter_sizes[1])
        self.conv3 = Conv1d(embedding_size, feature_map_size, filter_sizes[2])
        self.linear = Linear(feature_map_size, 1)
        self.dropout = dropout

    def forward(self, embeddings):
        """Forward pass of the CNN model.

        Args:
            embeddings: Input tensor of shape [batch x sentence_length x embedding_dim]

        Returns:
            Tensor: Output probabilities of shape [batch]
        """
        inputs = embeddings.permute(0, 2, 1)

        conv_out1 = self.conv1(inputs).relu()
        conv_out2 = self.conv2(inputs).relu()
        conv_out3 = self.conv3(inputs).relu()

        pooled_out1 = minitorch.max(conv_out1, dim=2)
        pooled_out2 = minitorch.max(conv_out2, dim=2)
        pooled_out3 = minitorch.max(conv_out3, dim=2)

        combined_out = pooled_out1 + pooled_out2 + pooled_out3

        fc_out = self.linear(combined_out.view(combined_out.shape[0], self.feature_map_size))
        fc_out = minitorch.dropout(fc_out, self.dropout, not self.training)

        return fc_out.sigmoid().view(fc_out.shape[0])


def get_predictions_array(y_true, model_output):
    """Convert model outputs to prediction arrays.

    Args:
        y_true: Ground truth labels
        model_output: Model predictions

    Returns:
        list: List of tuples containing (true_label, predicted_label, logit)
    """
    predictions_array = []
    for j, logit in enumerate(model_output.to_numpy()):
        true_label = y_true[j]
        if logit > 0.5:
            predicted_label = 1.0
        else:
            predicted_label = 0
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array


def get_accuracy(predictions_array):
    """Calculate accuracy from predictions array.

    Args:
        predictions_array: List of (true_label, predicted_label, logit) tuples

    Returns:
        float: Accuracy score between 0 and 1
    """
    correct = 0
    for y_true, y_pred, logit in predictions_array:
        if y_true == y_pred:
            correct += 1
    return correct / len(predictions_array)


best_val = 0.0


def default_log_fn(
    epoch,
    train_loss,
    losses,
    train_predictions,
    train_accuracy,
    validation_predictions,
    validation_accuracy,
):
    """Default logging function for training progress.

    Args:
        epoch: Current epoch number
        train_loss: Training loss for current epoch
        losses: List of historical losses
        train_predictions: Training predictions
        train_accuracy: List of training accuracies
        validation_predictions: Validation predictions
        validation_accuracy: List of validation accuracies
    """
    global best_val
    best_val = (
        best_val if best_val > validation_accuracy[-1] else validation_accuracy[-1]
    )
    print(f"Epoch {epoch}, loss {train_loss}, train accuracy: {train_accuracy[-1]:.2%}")
    if len(validation_predictions) > 0:
        print(f"Validation accuracy: {validation_accuracy[-1]:.2%}")
        print(f"Best Valid accuracy: {best_val:.2%}")


class SentenceSentimentTrain:
    """Training class for sentiment classification models."""

    def __init__(self, model):
        """Initialize trainer.

        Args:
            model: Model to train
        """
        self.model = model

    def train(
        self,
        data_train,
        learning_rate,
        batch_size=10,
        max_epochs=500,
        data_val=None,
        log_fn=default_log_fn,
    ):
        """Train the model.

        Args:
            data_train: Training data tuple (X, y)
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs
            data_val: Optional validation data tuple (X, y)
            log_fn: Logging function for training progress
        """
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, batch_size)
            ):
                y = minitorch.tensor(
                    y_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x)
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -(prob.log() / y.shape[0]).sum()
                loss.view(1).backward()

                # Save train predictions
                train_predictions += get_predictions_array(y, out)
                total_loss += loss[0]

                # Update
                optim.step()

            # Evaluate on validation set at the end of the epoch
            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()
                y = minitorch.tensor(
                    y_val,
                    backend=BACKEND,
                )
                x = minitorch.tensor(
                    X_val,
                    backend=BACKEND,
                )
                out = model.forward(x)
                validation_predictions += get_predictions_array(y, out)
                validation_accuracy.append(get_accuracy(validation_predictions))
                model.train()

            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss)
            log_fn(
                epoch,
                total_loss,
                losses,
                train_predictions,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )
            total_loss = 0.0


def encode_sentences(
    dataset, N, max_sentence_len, embeddings_lookup, unk_embedding, unks
):
    """Encode sentences using word embeddings.

    Args:
        dataset: Dataset containing sentences
        N: Number of sentences to encode
        max_sentence_len: Maximum sentence length for padding
        embeddings_lookup: Word embeddings lookup
        unk_embedding: Embedding vector for unknown words
        unks: Set to track unknown words

    Returns:
        tuple: (X, y) where X contains encoded sentences and y contains labels
    """
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        # pad with 0s to max sentence length in order to enable batching
        # TODO: move padding to training code
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                # use random embedding for unks
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)

    # load labels
    ys = dataset["label"][:N]
    return Xs, ys


def encode_sentiment_data(dataset, pretrained_embeddings, N_train, N_val=0):
    """Encode sentiment dataset using pretrained embeddings.

    Args:
        dataset: Raw sentiment dataset
        pretrained_embeddings: Pretrained word embeddings
        N_train: Number of training examples
        N_val: Number of validation examples

    Returns:
        tuple: ((X_train, y_train), (X_val, y_val)) containing encoded data
    """
    #  Determine max sentence length for padding
    max_sentence_len = 0
    for sentence in dataset["train"]["sentence"] + dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))

    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for i in range(pretrained_embeddings.d_emb)
    ]
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")

    return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
    train_size = 450
    validation_size = 100
    learning_rate = 0.01
    max_epochs = 250

    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=50, show_progress=True),
        train_size,
        validation_size,
    )
    model_trainer = SentenceSentimentTrain(
        CNNSentimentKim(feature_map_size=100, filter_sizes=[3, 4, 5], dropout=0.25)
    )
    model_trainer.train(
        (X_train, y_train),
        learning_rate,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
    )
