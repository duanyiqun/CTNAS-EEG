import os
import scipy.io as sio
import mne
import matplotlib.pyplot as plt
from mudus.visualization.visualize_utils_mne import raw_construct_mne_info


def load_raw_data(data_path, subject_id=1):
    output_path = os.path.join(data_path, 'cross_sub')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = '{}/{}{}{}'.format(output_path, 'cross_subject_data_', str(subject_id), '.mat')
    data = sio.loadmat(file_path)
    test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
    train_X = data["train_x"][:, :, 750:1500]
    test_y = data["test_y"].ravel()
    train_y = data["train_y"].ravel()
    return train_X, test_X, train_y, test_y


if __name__ == '__main__':
    train_X, test_X, train_y, test_y = load_raw_data('/Users/bytedance/Documents/BCI/Mudus_BCI/data/bci_iv/')
    custom_epochs = raw_construct_mne_info(test_X, test_y, visual_len=200)
    print(custom_epochs)
    # custom_epochs['Left'].average().plot(time_unit='s')
    # custom_epochs.plot_image(1, cmap='interactive', sigma=1., vmin=-250, vmax=250)
    # custom_epochs.plot(scalings='auto')
    events = mne.pick_events(custom_epochs.events, include=[1, 2, 3, 4])
    mne.viz.plot_events(events)
    plt.show()
    custom_epochs.plot(events=events, event_color={1: 'r', 2: 'b', 3: 'y', 4: 'g'}, title='Raw trail plot',
                       scalings=30, block=False, n_epochs=5, n_channels=11, epoch_colors=None, butterfly=False)
    # here epoch colors is with shape epochs *  channels
    plt.show()
    custom_epochs.plot_image(title='Epoch value time series wise', sigma=1., cmap='YlGnBu_r')    # here epoch colors is with shape epochs *  channels
    plt.show()

    mne.viz.plot_epochs_image(custom_epochs,cmap='interactive',title='Whole data heat map of epochs')
    # custom_epochs['Left'].average().plot(time_unit='s')
    # custom_epochs['Right'].average().plot(time_unit='s')

    evoked = custom_epochs.average()
    info = custom_epochs.info
    info.set_montage('standard_1020')
    print(evoked.data.shape)
    mne.viz.plot_topomap(evoked.data[:, 1], info, show=True)

    right_av = custom_epochs['Right'].average()
    left_av = custom_epochs['Left'].average()
    joint_kwargs = dict(ts_args=dict(time_unit='s'),
                        topomap_args=dict(time_unit='s'))
    right_av.plot_joint(title='Right averaged evoked data', show=True, **joint_kwargs)
    left_av.plot_joint(title='Left averaged evoked data', show=True, **joint_kwargs)
    plt.show()

    style_plot = dict(
        colors={"Left": "Crimson", "Right": "Cornflowerblue"},
        # linestyles={"Concrete": "-", "Abstract": ":"},
        split_legend=True,
        ci=.68,
        show_sensors='lower right',
        legend='lower left',
        truncate_yaxis="auto",
        picks=custom_epochs.ch_names.index("Pz"), )
    fig, ax = plt.subplots(figsize=(6, 4))
    evokeds = {'Left': custom_epochs['Left'].average(), 'Right': custom_epochs['Right'].average()}
    mne.viz.plot_compare_evokeds(evokeds, axes=ax, **style_plot)
    plt.show()


