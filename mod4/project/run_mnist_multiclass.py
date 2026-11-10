from mnist import MNIST
import minitorch

mndata = MNIST("project/data/")
images, labels = mndata.load_training()

BACKEND = minitorch.TensorBackend(minitorch.FastOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28

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

class Conv2d(minitorch.Module):
    """2D convolutional layer.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kh: Kernel height
        kw: Kernel width
    """
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        """Forward pass of conv2d layer.

        Args:
            input: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Tensor: Output tensor of shape (batch, out_channels, height-kh+1, width-kw+1)
        """
        out = minitorch.conv2d(input, self.weights.value) + self.bias.value
        return out

class Network(minitorch.Module):
    """Neural network for MNIST digit classification.

    The network consists of:
    - Two convolutional layers with ReLU activation
    - Average pooling
    - Two fully connected layers with dropout
    - Log softmax output
    """

    def __init__(self):
        super().__init__()

        # For visualization
        self.mid_features = None
        self.final_features = None

        # Define the layers
        self.conv1 = Conv2d(1, 4, 3, 3)  # First convolutional layer
        self.conv2 = Conv2d(4, 8, 3, 3)  # Second convolutional layer
        self.fc1 = Linear(392, 64)       # First fully connected layer
        self.fc2 = Linear(64, C)         # Second fully connected layer

    def forward(self, inputs):
        """Forward pass of the network.

        Args:
            inputs: Input tensor of shape (batch, channels, height, width)

        Returns:
            Tensor: Output tensor of shape (batch, num_classes)
        """
        # First convolutional block
        conv1_out = self.conv1(inputs).relu()
        self.mid_features = conv1_out  # Save intermediate features for visualization

        # Second convolutional block
        conv2_out = self.conv2(conv1_out).relu()
        self.final_features = conv2_out  # Save final features before pooling for visualization

        # Apply average pooling
        pooled_out = minitorch.avgpool2d(conv2_out, (4, 4))

        # Flatten for fully connected layers
        batch_size = pooled_out.shape[0]
        flattened = pooled_out.view(batch_size, 392)

        # Fully connected layers
        fc1_out = self.fc1(flattened).relu()
        dropout_out = minitorch.dropout(fc1_out, 0.25, not self.training)
        logits = self.fc2(dropout_out)

        # Log softmax for output probabilities
        output = minitorch.logsoftmax(logits, dim=1)
        return output

def make_mnist(start, stop):
    """Create MNIST dataset tensors.

    Args:
        start: Starting index in MNIST dataset
        stop: Ending index in MNIST dataset

    Returns:
        tuple: (X, y) where X is list of image arrays and y is list of one-hot label vectors
    """
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys

def default_log_fn(epoch, total_loss, correct, total, losses, model):
    """Default logging function for training progress.

    Args:
        epoch: Current epoch number
        total_loss: Total loss for current epoch
        correct: Number of correct predictions
        total: Total number of predictions
        losses: List of losses
        model: Current model
    """
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")

class ImageTrain:
    """Training class for MNIST image classification.

    Handles model creation, training loop, and evaluation.
    """
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        """Run inference on a single input.

        Args:
            x: Input image tensor

        Returns:
            Tensor: Model predictions
        """
        return self.model.forward(minitorch.tensor([x], backend=BACKEND))

    def train(self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn):
        """Train the model.

        Args:
            data_train: Training data tuple (X_train, y_train)
            data_val: Validation data tuple (X_val, y_val)
            learning_rate: Learning rate for optimization
            max_epochs: Maximum number of training epochs
            log_fn: Logging function for training progress
        """
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            model.train()

            for batch_num, example_num in enumerate(range(0, n_training_samples, BATCH)):
                if n_training_samples - example_num <= BATCH:
                    continue

                y = minitorch.tensor(y_train[example_num : example_num + BATCH], backend=BACKEND)
                x = minitorch.tensor(X_train[example_num : example_num + BATCH], backend=BACKEND)
                x.requires_grad_(True)
                y.requires_grad_(True)

                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                optim.step()

                if batch_num % 5 == 0:
                    model.eval()
                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(y_val[val_example_num : val_example_num + BATCH], backend=BACKEND)
                        x = minitorch.tensor(X_val[val_example_num : val_example_num + BATCH], backend=BACKEND)
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)

                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model)
                    total_loss = 0.0
                    model.train()

if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.01)
