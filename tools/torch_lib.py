import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import datashader as ds
import pandas as pd

activations = {
    'relu': nn.ReLU(),
    'leaky-relu': nn.LeakyReLU(),
    'softmax': nn.Softmax(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'siLU': nn.SiLU()
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


def get_standard_scaler_transform(data):
    def forward(data_): return (data_ - np.mean(data)) / np.std(data)

    def backward(data_): return data_ * np.std(data) + np.mean(data)

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
    def __init__(self, df_, model, model_output_transformer, x_attribute, y_attribute, input_attributes,
                 output_attribute):
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

    def plot_single_graphic(self, y_v, x_v, show_legend: bool):
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

        y_actual = data_df[self.output_attribute].to_numpy()  # use backward transform for actual data
        self.transformer.set_data(y_actual)
        y_actual = self.transformer.transform_backward()

        return go.Scatter(x=x, y=y_actual, marker=dict(color='blue'), legendgroup='1', name='actual',
                          showlegend=show_legend), \
               go.Scatter(x=x, y=y_pred, marker=dict(color='red'), legendgroup='1', name='predicted',
                          showlegend=show_legend)  # actual, predicted

    def plot_subject_graphic(self):
        graphic = make_subplots(rows=len(self.y_attribute_values),
                                cols=len(self.x_attribute_values),
                                start_cell="top-left",
                                column_titles=[self.x_attribute + "=" + str(np.round(x_val, 4)) for x_val in
                                               self.x_attribute_values],
                                row_titles=[self.y_attribute + "=" + str(np.round(y_val, 4)) for y_val in
                                            self.y_attribute_values])
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


def remove_outliers(data, cut_factor: float = 4.0):
    quantile_third = torch.quantile(data, 0.75)
    quantile_first = torch.quantile(data, 0.25)
    delta = quantile_third - quantile_first

    threshold = quantile_third + delta * cut_factor

    result = data[data < threshold]
    outliers_count = len(data) - len(result)

    return result, outliers_count


# plot relative errors using datashader
def plot_relative_error_shader(y_true, y_pred, target_error: float, title: str, width, height, drop_outliers=False):
    assert len(y_true) == len(y_pred)
    data = 100 * torch.abs((y_true - y_pred)) / y_true  # percent
    target_error *= 100
    canvas = ds.Canvas(plot_width=width, plot_height=height)

    outliers_percent = 0

    if drop_outliers:
        data_, outliers_count = remove_outliers(data)
        outliers_percent = round(100 * outliers_count / len(data), 4)
        data = data_

    x = np.arange(1, len(y_true) + 1)
    y = data.to('cpu').numpy()
    source = pd.DataFrame({'x': x, 'y': y})
    coordinates = canvas.points(source, x='x', y='y')

    zero_mask = coordinates.values == 0
    coordinates.values = np.log10(coordinates.values, where=np.logical_not(zero_mask))
    coordinates.values[zero_mask] = np.nan

    # plot data
    fig = px.imshow(coordinates, origin='lower', labels={'color': 'Log10(count)'})
    fig.update_traces(hoverongaps=False)
    fig.update_layout(coloraxis_colorbar=dict(title='Count', tickprefix='1.e'))

    result = go.Figure()
    result.add_trace(go.Heatmap(fig.data[0]))

    # add threshold:
    st_point = x[0]
    end_point = x[len(x) - 1]
    result.add_trace(
        go.Scatter(x=[st_point, end_point], y=[target_error, target_error], mode='lines', name='threshold'))
    result.add_trace(
        go.Scatter(x=[st_point, end_point], y=[-target_error, -target_error], mode='lines', name='threshold'))

    inside_interval_percent = round(100 * len(data[torch.abs(data) <= target_error]) / len(data), 4)
    title_text = title + f" (inside interval percent = {inside_interval_percent}%)"

    if drop_outliers:
        title_text += f"; outliers percent={outliers_percent}%"

    result.update_layout(title=title_text, height=height, width=width, template="plotly_white",
                         xaxis_title='predictions', yaxis_title='error %')

    return result


def plot_relative_error(y_true, y_pred, target_error: float, title: str, width, height):
    assert len(y_true) == len(y_pred)
    data = torch.abs((y_true - y_pred)) / y_true
    x = np.arange(1, len(y_true) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=data, mode='markers', name='error distribution'))

    st_point = x[0]
    end_point = x[len(x) - 1]
    fig.add_trace(go.Scatter(x=[st_point, end_point], y=[target_error, target_error], mode='lines', name='threshold'))
    fig.add_trace(go.Scatter(x=[st_point, end_point], y=[-target_error, -target_error], mode='lines', name='threshold'))

    inside_interval_percent = round(100 * len(data[torch.abs(data) <= target_error]) / len(data), 4)

    fig.update_layout(title=title + f" (inside interval percent = {inside_interval_percent}%)",
                      height=height, width=width)

    return fig


def plot_relative_error_hist(y_true, y_pred, target_error: float, title: str,
                             bins_count: int = 100, drop_outliers=True):
    assert len(y_true) == len(y_pred)
    data = 100 * torch.abs((y_true - y_pred)) / y_true  # error in percent
    outliers_percent = 0
    target_error *= 100  # percent

    if drop_outliers:
        data_, outliers_count = remove_outliers(data)
        outliers_percent = round(100 * outliers_count / len(data), 4)
        data = data_

    min_val = torch.min(data)
    max_val = torch.max(data)

    hist_config = dict(start=min_val,
                       end=max_val,
                       size=(max_val - min_val) / bins_count
                       )
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, xbins=hist_config))
    fig.add_vline(x=target_error, line_dash="dash", line_color='red')

    inside_interval_percent = round(100 * len(data[torch.abs(data) <= target_error]) / len(data), 4)
    title_text = title + f" (inside interval percent = {inside_interval_percent}%)"

    if drop_outliers:
        title_text += f"; outliers percent={outliers_percent}%"

    fig.update_layout(title=title_text, xaxis_title='error %', yaxis_title='predictions count',
                      height=500, width=1500, template='plotly_white')

    return fig


def plot_relative_errors(outputs, full_dataset_loader,
                         model, transform_dict, df_=None,
                         threshold: float = 0.05, device: str = 'cuda',
                         plots_dir: str = "", width: int = 10000,
                         height: int = 500, with_shader=True,
                         mode: str = 'default', bin_count: int = None):
    def save_and_show_fig(fig, fig_name: str):
        if fig is not None:
            fig.show('browser')
            fig.write_image(fig_name)

    predictions, actuals = predict(full_dataset_loader, model, device)

    for i in range(len(outputs)):
        output_attr = outputs[i]
        transformer = transform_dict[output_attr]

        # transform predictions back:
        prediction = predictions[:, i]
        transformer.set_data_from_tensor(prediction)
        final_prediction = torch.from_numpy(transformer.transform_backward())

        if df_ is None:
            # get actual data from dataloader
            actual = actuals[:, i]
            transformer.set_data_from_tensor(actual)
            actual = torch.from_numpy(transformer.transform_backward())
        else:
            # get actual data from source dataframe:
            actual = torch.from_numpy(df_[output_attr].to_numpy())

        # plot graphic
        modes = set(mode.split('+'))
        default_fig = None
        hist_fig = None

        if "default" in modes:
            if with_shader:
                default_fig = plot_relative_error_shader(actual, final_prediction, threshold,
                                                         'relative error ' + output_attr,
                                                         width, height)
            else:
                default_fig = plot_relative_error(actual, final_prediction, threshold,
                                                  'relative error ' + output_attr,
                                                  width, height)
        if "hist" in modes:
            if bin_count is None:
                assert False and "No bin_count defined for histogram mode"
            hist_fig = plot_relative_error_hist(actual, final_prediction, threshold,
                                                'relative error hist ' + output_attr)

        # save figs (hist and default) if exists
        save_and_show_fig(default_fig, plots_dir + "relative_error_" + output_attr + ".pdf")
        save_and_show_fig(hist_fig, plots_dir + "relative_error_hist_" + output_attr + ".pdf")


def plot_layer_distribution(model, layer_idx: int, batch, bins_count: int = 100):
    layers_output_dict = dict()

    def get_activation(layer_name: str):
        def get_output_hook(module, input, output):
            layers_output_dict[layer_name] = output.detach()

        return get_output_hook  # return hook as function

    layer_name = 'layer-' + str(layer_idx)
    handler = model.activations[layer_idx].register_forward_hook(get_activation(layer_name))

    with torch.no_grad():
        model(batch)  # do forward prop

    handler.remove()  # remove hook
    layer_output = layers_output_dict[layer_name]
    layer_output = layer_output.flatten()

    max_val = torch.max(layer_output).item()
    min_val = torch.min(layer_output).item()
    print(min_val)
    bin_config = dict(start=min_val,
                      end=max_val,
                      size=(max_val - min_val) / bins_count
                      )
    fig = go.Figure(data=[go.Histogram(x=layer_output.to('cpu').numpy(), xbins=bin_config)])
    fig.show()


class Predictor(object):
    def __init__(self, dataloader, actual_df, transform_dict, model, inputs, outputs):
        self.dataloader_ = dataloader
        self.transform_dict_ = transform_dict
        self.model_ = model
        self.df_ = actual_df
        self.inputs_ = inputs
        self.outputs_ = outputs

    def predict(self, device='cuda'):
        predictions, _ = predict(self.dataloader_, self.model_, device)
        predictions_dict = dict()
        actuals_dict = dict()

        for i in range(len(self.outputs_)):
            output_attr = self.outputs_[i]
            transformer = self.transform_dict_[output_attr]

            # transform predictions back:
            prediction = predictions[:, i]
            transformer.set_data_from_tensor(prediction)
            final_prediction = torch.from_numpy(transformer.transform_backward())

            predictions_dict[output_attr] = final_prediction
            actuals_dict[output_attr] = self.df_[output_attr].to_numpy()

        return predictions_dict, actuals_dict


def plot_predictions(outputs, full_dataset_loader, model, plots_dir: str = "", device: str = 'cuda'):
    predictions, actuals = predict(full_dataset_loader, model, device)
    assert predictions.size() == actuals.size()

    for i in range(len(outputs)):
        output_attr = outputs[i]

        prediction = predictions[:, i]
        actual = actuals[:, i]
        approximation_graphic = plot_tensor_approximation(actual, prediction, output_attr, 'lines+markers', 12000, 900)

        approximation_graphic.show('browser')
        approximation_graphic.write_image(plots_dir + "approximation_" + output_attr + ".pdf")


def plot_actual_predictions(outputs, full_dataset_loader,
                            model, transform_dict, df_=None,
                            device: str = 'cuda', plots_dir: str = "",
                            width: int = 12000, height: int = 900):
    predictions, actuals = predict(full_dataset_loader, model, device)

    for i in range(len(outputs)):
        output_attr = outputs[i]
        transformer = transform_dict[output_attr]

        # transform predictions back:
        prediction = predictions[:, i]
        transformer.set_data_from_tensor(prediction)
        final_prediction = torch.from_numpy(transformer.transform_backward())

        if df_ is None:
            # get actual data from dataloader
            actual = actuals[:, i]
            transformer.set_data_from_tensor(actual)
            actual = torch.from_numpy(transformer.transform_backward())
        else:
            # get actual data from source dataframe:
            actual = torch.from_numpy(df_[output_attr].to_numpy())
        # plot graphics:
        approximation_graphic = plot_tensor_approximation(actual, final_prediction, output_attr, 'lines+markers', width,
                                                          height)
        approximation_graphic.show('browser')
        approximation_graphic.write_image(plots_dir + "approximation_real_" + output_attr + ".pdf")


def train_loop(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    epoch_loss = 0
    num_batches = len(dataloader)

    def train_step(X, y):
        optimizer.zero_grad()  # zero gradients to prevent accumulated gradient value

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        return loss.item()

    for _, (X, y) in enumerate(dataloader):
        epoch_loss += train_step(X, y)

    epoch_loss = epoch_loss / num_batches  # get average loss

    return epoch_loss


def predict(dataloader, model, device):
    model.eval()  # inference - set to eval
    predictions = torch.empty(0).to(device)
    actual = torch.empty(0).to(device)

    with torch.no_grad():  # we are making predictions, so it's no need to calc gradients
        for _, (X, y) in enumerate(dataloader):
            pred = model(X)
            assert y.size() == pred.size()

            predictions = torch.concat((predictions, pred), 0)
            actual = torch.concat((actual, y), 0)

    return predictions, actual


def test_loop(dataloader, model, loss_fn):
    model.eval()  # inference - set to eval
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():  # we test model, so it's no need to calc gradients
        for _, (X, y) in enumerate(dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

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
