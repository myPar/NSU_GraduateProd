import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

activations = {
    'relu': nn.ReLU(),
    'softmax': nn.Softmax(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh()
}


def plot_loss(loss_list, graphic_name):
    # plot loss graphic
    x = np.arange(0, len(loss_list))
    y = np.array(loss_list)

    assert len(x) == len(y)

    fig = px.line(x=x, y=y, labels={'x': 'epoch', 'y': 'loss'}, title=graphic_name)
    fig.show()


class TimeDistributedLayer(nn.Module):
    def __init__(self, batch_first: bool, module):
        super().__init__()
        self.batch_first = batch_first
        self.module_block = module

    def forward(self, x):
        if len(x.size()) <= 2:  # there is batched input (without timestamps) or only one vector
            return self.module_block(x)

        y = x.contiguous().view(-1, x.size(-1))  # shape = (batch_size * timestamps, out_size)
        y = self.module_block(y)

        # reshape output tensor properly!
        if self.batch_first:
            y = y.view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))

        return y


def plot_tensor_approximation(actual, predictions, attribute_name, mode: str, width, height):
    assert actual.size() == predictions.size()
    comparison_graphic = go.Figure()

    x = np.arange(0, len(actual))
    y_actual = actual.cpu().numpy()
    y_predicted = predictions.cpu().numpy()

    comparison_graphic.add_trace(go.Scatter(x=x, y=y_actual,
                                            mode=mode,
                                            name=attribute_name + ' actual graphic', ))

    comparison_graphic.add_trace(go.Scatter(x=x, y=y_predicted,
                                            mode=mode,
                                            name=attribute_name + ' predicted', ))

    comparison_graphic.update_layout(
        height=height,
        width=width,
        title_text=attribute_name
    )

    return comparison_graphic


def normalize(data, interval):
    data_min = data.min()
    data_max = data.max()

    return ((data - data_min) / (data_max - data_min)) * (interval[1] - interval[0]) + interval[0]


def get_normalize_transforms(data, interval):
    def forward(data_): return normalize(data_, interval)
    def backward(data_): return normalize(data_, [data.min(), data.max()])

    return forward, backward


class AttributeTransformer(object):
    def __init__(self, attribute_data):
        self.data = attribute_data
        self.backward_transforms_list = list()

    def transform(self, forward, backward):
        self.data = forward(self.data)
        self.backward_transforms_list.insert(0, backward)

        return self.data

    def set_data_from_tensor(self, tensor_data):
        self.data = tensor_data.cpu().numpy().flatten()

    def set_data(self, data_):
        self.data = data_.flatten()

    def drop_transform(self, idx):
        assert 0 <= idx < len(self.backward_transforms_list)
        self.backward_transforms_list.pop(idx)

    def transform_backward(self):
        for transform in self.backward_transforms_list:
            self.data = transform(self.data)

        return self.data


class SubjectGraphicPlotter(object):
    def __init__(self, df_, model, model_output_transformer, x_attribute, y_attribute, input_attributes, output_attribute):
        assert x_attribute in df_.columns and y_attribute in df_.columns
        self.x_attribute = x_attribute
        self.y_attribute = y_attribute
        self.df = df_
        self.model = model
        self.transformer = model_output_transformer
        self.x_attribute_values = df_[x_attribute].unique()
        self.y_attribute_values = df_[y_attribute].unique()
        self.data_dict = self.build_graphics_data_dict()

        self.inputs = input_attributes
        self.output_attribute = output_attribute

        self.single_width = 100
        self.single_height = 80

    def set_model(self, model):
        self.model = model

    def set_single_dim(self, width, height):
        self.single_width = width
        self.single_height = height

    def build_graphics_data_dict(self):
        data_dict = dict()
        df_ = self.df

        for y_v in self.y_attribute_values:
            data_dict[y_v] = dict()
        for y_v in self.y_attribute_values:
            for x_v in self.x_attribute_values:
                data_dict[y_v][x_v] = df_.loc[(df_[self.x_attribute] == x_v) & (df_[self.y_attribute] == y_v)]

        return data_dict

    def make_titles(self):
        titles = list()

        for y_v in self.y_attribute_values:
            for x_v in self.x_attribute_values:
                titles.append(f"{self.y_attribute}={np.round(y_v, 4)} {self.x_attribute}={np.round(x_v, 4)}")
        return tuple(titles)

    # predict samples one by one
    def get_partial_data_prediction(self, input_tensor, model_device):
        result = torch.tensor([]).to(model_device)

        for sample in input_tensor:
            sample = torch.unsqueeze(sample, 0)
            output = self.model(sample)
            result = torch.cat((result, output), 0)

        return result

    def plot_single_graphic(self, y_v, x_v, show_legend:bool):
        data_df = self.data_dict[y_v][x_v]
        model_device = next(self.model.parameters()).device

        if data_df.shape[0] == 0:
            return go.Scatter(), go.Scatter()

        with torch.no_grad():
            input_tensor = torch.from_numpy(data_df[self.inputs].to_numpy()).float().to(model_device)

            output = self.get_partial_data_prediction(input_tensor, model_device)
            self.transformer.set_data_from_tensor(torch.flatten(output))

        x = np.arange(0, len(output))

        y_pred = self.transformer.transform_backward()  # use backward transform for predictions

        y_actual = data_df[self.output_attribute].to_numpy()   # use backward transform for actual data
        self.transformer.set_data(y_actual)
        y_actual = self.transformer.transform_backward()

        return go.Scatter(x=x, y=y_actual, marker=dict(color='blue'), legendgroup='1', name='actual', showlegend=show_legend), \
               go.Scatter(x=x,y=y_pred, marker=dict(color='red'), legendgroup='1', name='predicted', showlegend=show_legend)    # actual, predicted

    def plot_subject_graphic(self):
        graphic = make_subplots(rows=len(self.y_attribute_values),
                                cols=len(self.x_attribute_values),
                                start_cell="top-left",
                                column_titles=[self.x_attribute + "=" +str(np.round(x_val, 4)) for x_val in self.x_attribute_values],
                                row_titles=[self.y_attribute + "=" + str(np.round(y_val, 4)) for y_val in self.y_attribute_values])
        row, col = 1, 1
        show_legend = True

        for y_v in self.y_attribute_values:
            col = 1
            for x_v in self.x_attribute_values:
                actual_graphic, predicted_graphic = self.plot_single_graphic(y_v, x_v, show_legend)
                show_legend = False

                graphic.add_trace(actual_graphic, row=row, col=col)
                graphic.add_trace(predicted_graphic, row=row, col=col)

                col += 1
            row += 1

        graphic.update_layout(height=self.single_height * len(self.y_attribute_values),
                              width=self.single_width * len(self.x_attribute_values), template='plotly_white')
        return graphic


def plot_relative_error(y_true, y_pred, target_error: float, title: str, width, height):
    assert len(y_true) == len(y_pred)
    data = torch.abs((y_true - y_pred)) / y_true
    x = np.arange(1, len(y_true))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=data, mode='markers', name='error distribution'))

    st_point = x[0]
    end_point = x[len(x) - 1]
    fig.add_trace(go.Scatter(x=[st_point, end_point], y=[target_error, target_error], mode='lines', name='threshold'))
    fig.add_trace(go.Scatter(x=[st_point, end_point], y=[-target_error, -target_error], mode='lines', name='threshold'))

    inside_interval_percent = len(data[torch.abs(data) <= target_error]) / len(data)
    fig.update_layout(title=title + " (inside interval share = " + str(inside_interval_percent) + ")", height=height, width=width)

    return fig


def train_loop(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    epoch_loss = 0
    num_batches = len(dataloader)

    def train_step(X, y):
        optimizer.zero_grad()   # zero gradients to prevent accumulated gradient value

        # Compute prediction and loss
        pred = torch.flatten(model(X))
        loss = loss_fn(pred, torch.flatten(y))

        # Backpropagation
        loss.backward()
        optimizer.step()

        return loss.item()

    for _, (X, y) in enumerate(dataloader):
        epoch_loss += train_step(X, y)

    epoch_loss = epoch_loss / num_batches # get average loss

    return epoch_loss


def predict(dataloader, model, device):
    predictions = torch.empty(0).to(device)
    actual = torch.empty(0).to(device)

    with torch.no_grad(): # we are making predictions, so it's no need to calc gradients
        for _, (X, y) in enumerate(dataloader):
            pred = model(X)
            assert y.size() == pred.size()
            pred_flatten = torch.flatten(pred)
            y_flatten = torch.flatten(y)

            predictions = torch.concat((predictions, pred_flatten), 0)
            actual = torch.concat((actual, y_flatten), 0)

    return predictions, actual


def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad(): # we test model, so it's no need to calc gradients
        for _, (X, y) in enumerate(dataloader):
            pred = torch.flatten(model(X))
            test_loss += loss_fn(pred, torch.flatten(y)).item()

    test_loss = test_loss / num_batches

    return test_loss


def train_model(epoch_count, model, optimizer, loss_function,
                train_loader, test_loader, epoch_validation: bool, train_loss_threshold: float):
    train_loss_list = list()
    validation_loss_list = list()

    for epoch_count in range(epoch_count):
        train_loss = train_loop(train_loader, model, loss_function, optimizer)
        train_loss_list.append(train_loss)

        if epoch_validation:
            test_loss = test_loop(test_loader, model, loss_function)
            validation_loss_list.append(test_loss)
            print(f"Epoch: {epoch_count}; train loss={train_loss:>8f}; validation loss={test_loss:>8f}")
        else:
            print(f"Epoch: {epoch_count}; train loss={train_loss:>8f}")

        if train_loss < train_loss_threshold:
            break

    return train_loss_list, validation_loss_list
