import pandas as pd
from tools.torch_lib import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import copy
from torchmetrics.regression import MeanAbsolutePercentageError


class SimpleDataset(Dataset):
    def __init__(self, df_, inputs, outputs, device):
        self.df = df_
        self.inputs = torch.from_numpy(df_[inputs].to_numpy()).float().to(device)
        self.outputs = torch.from_numpy(df_[outputs].to_numpy()).float().to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item, label = self.inputs[idx], self.outputs[idx]

        return item, label


class LinearModel(nn.Module):
    def __init__(self, layers_dims, act_str_list, output_dim):
        super().__init__()
        layers_count = len(layers_dims)
        assert layers_count > 0

        module_list = []
        for i in range(layers_count - 1):
            module_list.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        module_list.append(nn.Linear(layers_dims[layers_count - 1], output_dim))

        activations_list = []
        for i in range(layers_count):
            activations_list.append(activations[act_str_list[i]])

        self.linears = nn.ModuleList(module_list)
        self.activations = nn.ModuleList(activations_list)

    def forward(self, x):
        y = x

        for lin, act in zip(self.linears, self.activations):
            y = lin(y)
            y = act(y)

        return y


interval_th = [-1, 1]  # normalization interval for 'th' activation function
interval_sigmoid = [0, 1]  # normalization interval for 'sigmoid' activation function


def run_baseline(batch_size, learning_rate, epoch_count, train_loss_threshold):
    gpu = torch.device('cuda')
    cpu = torch.device('cpu')
    device = cpu

    if torch.cuda.is_available():
        device = gpu
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

    print(f"attached device - {device}")

    ### Load dataframe
    dataset_dir = "dataset/"
    dataset_file_name = "1D_2L_chart.csv"
    plots_dir = "plots/"
    df = pd.read_csv(dataset_dir + dataset_file_name)

    # print attribute's min max
    print("print attribute's min max:")
    print(f"AO/d: min={df['AO/d'].min()} max={df['AO/d'].max()}")
    print(f"ro_formation: min={df['ro_formation'].min()} max={df['ro_formation'].max()}")
    print(f"lambda: min={df['lambda'].min()} max={df['lambda'].max()}")
    print(f"rok: min={df['rok'].min()} max={df['rok'].max()}")

    # resistance min max in logarithmic scale:
    print(f"AO/d: min={np.log(df['AO/d'].min())} max={np.log(df['AO/d'].max())}")
    print(f"ro_formation: min={np.log(df['ro_formation'].min())} max={np.log(df['ro_formation'].max())}")
    print(f"rok: min={np.log(df['rok'].min())} max={np.log(df['rok'].max())}")

    ### Add dataframe transforms
    inputs = np.array(['AO/d', 'lambda', 'ro_formation'])
    outputs = np.array(['rok'])

    logarithmic_columns = ['ro_formation', 'rok', 'AO/d']

    # normalize data ('min/max' normalization):
    normalize_interval = interval_sigmoid

    df_transformed = df.copy()
    rok_attr_transformer = AttributeTransformer(df_transformed[outputs].to_numpy())

    # transform 'rok':
    forward, backward = np.log, np.exp
    df_transformed['rok'] = rok_attr_transformer.transform(forward, backward)
    forward, backward = get_normalize_transforms(rok_attr_transformer.data, normalize_interval)
    df_transformed['rok'] = rok_attr_transformer.transform(forward, backward)

    # logarithm resistance:
    for col in logarithmic_columns:
        if col == 'rok':
            continue
        df_transformed[col] = df_transformed[col].apply(np.log)

    # add normalization
    for attribute in df_transformed.columns:
        if attribute == 'rok':
            continue
        transform, _ = get_normalize_transforms(df_transformed[attribute].to_numpy(), normalize_interval)
        df_transformed[attribute] = transform(df_transformed[attribute].to_numpy())

    ### Build Datasets and create dataloaders
    train_df, test_df = train_test_split(df_transformed, shuffle=True, test_size=0.3)

    train_dataset = SimpleDataset(train_df, inputs, outputs, device)
    test_dataset = SimpleDataset(test_df, inputs, outputs, device)
    full_dataset = SimpleDataset(df_transformed, inputs, outputs, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    full_dataset_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    ### Build model
    layers_dims = [len(inputs), 30, 100, 700, 100, 30, len(outputs)]
    layers_count = len(layers_dims)
    activations_string_list = ['relu'] * layers_count

    linear_model = LinearModel(layers_dims, activations_string_list, len(outputs)).to(device)

    optimizer = torch.optim.Adam(linear_model.parameters(), lr=learning_rate)
    loss_function = nn.L1Loss()
    epoch_validation = True

    ### Train model
    print("train model...")
    train_loss_list, validation_loss_list = train_model(epoch_count, linear_model,
                                                        optimizer, loss_function,
                                                        train_loader, test_loader,
                                                        epoch_validation, train_loss_threshold)
    plot_loss(train_loss_list, "train loss")
    test_loss = test_loop(test_loader, linear_model, loss_function)
    print(f"test loss={test_loss}")
    plot_loss(validation_loss_list, "test loss")

    ### Plot predictions
    predictions, actual = predict(full_dataset_loader, linear_model, device)
    print(predictions)  ######
    print(actual)       ######
    assert predictions.size() == actual.size()

    approximation_graphic = plot_tensor_approximation(actual, predictions, 'rok', 'lines+markers', 12000, 900)
    approximation_graphic.show()
    approximation_graphic.write_image(plots_dir + "pytorch_linear_approximation.pdf")

    ### Linear model final approximation
    rok_attr_transformer.set_data_from_tensor(predictions)
    predictions = torch.tensor(rok_attr_transformer.transform_backward())

    rok_attr_transformer.set_data_from_tensor(actual)
    actual = torch.tensor(rok_attr_transformer.transform_backward())

    approximation_graphic = plot_tensor_approximation(actual, predictions, 'rok', 'lines+markers', 12000, 900)
    approximation_graphic.show()
    approximation_graphic.write_image(plots_dir + "pytorch_linear_approximation_real.pdf")

    ### plot subject graphic
    rok_attr_transformer_dropped = copy.deepcopy(rok_attr_transformer)
    rok_attr_transformer_dropped.drop_transform(1)

    subject_graphic_plotter = SubjectGraphicPlotter(df_transformed, linear_model, rok_attr_transformer_dropped,
                                                    'lambda', 'AO/d', inputs, 'rok')
    subject_graphic_plotter.set_single_dim(500, 400)
    subject_graphic = subject_graphic_plotter.plot_subject_graphic()
    subject_graphic.write_image(plots_dir + "linear_subject_graphic.pdf")

    #### plot relative error
    mape = MeanAbsolutePercentageError()
    print(f"mape={mape(predictions, actual)}")

    fig = plot_relative_error(actual, predictions, 0.05, 'linear relative error', 10000, 500)
    fig.show('browser')
    fig.write_image(plots_dir + "linear_relative_error.pdf")

    ### Save model
    linear_model.to(cpu)  # attach model to cpu before scripting and saving to prevent cuda meta information saved
    scripted_model = torch.jit.script(linear_model)
    model_name = "saved_models/" + "linear_" + str(round(test_loss, 7)).replace('.', '_')

    scripted_model.save(model_name + ".pt")  # save torch script model which compatible with pytorch c++ api
    torch.save(linear_model, model_name + ".pth")  # save model in python services specific format

    # check model scripted model and source model equality:
    # order AO/d, lambda, ro_formation
    print(scripted_model(torch.tensor([0.6, 0.362372, 0.04])))
    print(linear_model(torch.tensor([0.6, 0.362372, 0.04])))


if __name__ == '__main__':
    run_baseline(batch_size=800,
                 learning_rate=0.00002,
                 epoch_count=100,
                 train_loss_threshold=0.00025
                 )
