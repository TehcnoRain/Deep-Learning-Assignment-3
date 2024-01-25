import torch
from torch import nn

def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    """
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    predictions = model(X)
    loss = loss_fn(predictions, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X_train, y_train):
    """
    Train the model for the given number of epochs.
    Hint: use the train_iteration function.
    Hint 2: while working, you can use the print function to print the loss every 1000 epochs.
    Hint 3: you can use the previous_loss variable to stop the training when the loss is not changing much.
    """
    learning_rate = 0.0001  # Adjust the learning rate as needed
    num_epochs = 10000  # Adjust the number of epochs as needed
    input_features = X_train.shape[1]  # Extract the number of features from the input shape of X
    output_features = y_train.shape[1]  # Extract the number of features from the output shape of y
    model = create_linear_regression_model(input_features, output_features)

    loss_fn = nn.MSELoss()  # Use mean squared error loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    for epoch in range(1, num_epochs):
        loss = train_iteration(X_train, y_train, model, loss_fn, optimizer)
        if epoch % 100 == 0:  # Change this condition to stop the training when the loss is not changing much.
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')
        if abs(previous_loss - loss) < 1e-6:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')
            break
        previous_loss = loss.item()
        # This is a good place to print the loss every 1000 epochs.
    return model, loss