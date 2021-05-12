import glob
import os
import numpy as np
from utils.helper import make_path, load_dict
import cv2
import torch
from torch.nn.utils.rnn import pad_sequence

def load_fmri(fmri_dir, subject, ROI):
    """This function loads fMRI data into a numpy array for to a given ROI.

    Parameters
    ----------
    fmri_dir : str
        path to fMRI data.
    subject : int or str
        subject number if int, 'sub#num' if str
    ROI : str
        name of ROI.

    Returns
    -------
    np.array
        matrix of dimensions #train_vids x #repetitions x #voxels
        containing fMRI responses to train videos of a given ROI
    """
    ROI_data = load_dict(make_path(fmri_dir, subject, ROI))
    return ROI_data["train"]

def load_voxel_mask_wb(fmri_dir, subject):
    """ Load voxel mask for whole brain fMRI data.

    Parameters
    ----------
    fmri_dir : str
        path to fMRI data.
    subject : int or str
        subject number if int, 'sub#num' if str

    Returns
    -------
    np.array
        matrix of dimensions #train_vids x #repetitions x #voxels
        containing fMRI responses to train videos of a given ROI
    """
    ROI_data = load_dict(make_path(fmri_dir, subject, ROI))
    return ROI_data['voxel_mask']

def load_video(file, frame_skip_step=1):
    """This function takes a video file as input and returns
    an array of frames in numpy format.

    Parameters
    ----------
    file : str
        path to video file
    frame_skip_step : int
        take every `frame_skip_step`th frame

    Returns
    -------
    video: np.array
        shape: (1, #num_frames, height, width, 3)
    """
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((int(frameCount / frame_skip_step), frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frameCount and ret:
        fc += 1
        if fc % frame_skip_step == 0:
            (ret, buf[int((fc - 1) / frame_skip_step)]) = cap.read()

    cap.release()
    return np.expand_dims(buf, axis=0)

def load_activations(activations_dir, layer_name):
    """This function loads neural network features/activations (preprocessed using PCA) into a
    numpy array according to a given layer.

    Parameters
    ----------
    activations_dir : str
        Path to PCA processed Neural Network features
    layer_name : str
        which layer of the neural network to load,

    Returns
    -------
    train_activations : np.array
        matrix of dimensions #train_vids x #pca_components
        containing activations of train videos
    test_activations : np.array
        matrix of dimensions #test_vids x #pca_components
        containing activations of test videos
    """

    train_file = os.path.join(activations_dir,"train_" + layer_name + ".npy")
    test_file = os.path.join(activations_dir,"test_" + layer_name + ".npy")
    train_activations = np.load(train_file)
    test_activations = np.load(test_file)
    scaler = StandardScaler()
    train_activations = scaler.fit_transform(train_activations)
    test_activations = scaler.fit_transform(test_activations)

    return train_activations, test_activations

def __pad_sequence(sequences, from_beginning=True, pad_value=0):
    """ Pad sequences with `pad_value` to the same length, which is chosen to be max length of all sequences. Padding is done `from_beginning` or not.

    Parameters
    ----------
    sequence : iterable
        Tensors representing sequences, first dimension is seq len, remaining dimensions and dtype must be the same for all elemenets.
    from_beginning : bool
        Wherher to add additional elements to the beginning or end of a sequence.
    pad_value : 
        A value used for padding.

    Returns
    -------
    padded_sequences : torch.tensor
        Tensor with padded sequences of shape (len(sequences), max_seq_len, ...)
    """
    if from_beginning is False:
        return pad_sequence(sequence, batch_first=True, padding_value=pad_value)

    if any([s.shape[1:] != sequences[0].shape[1:] for s in sequences]):
        raise ValueError("All sequences must have the same shape (except the first dimensions, 0).")

    if any([s.dtype != sequences[0].dtype for s in sequences]):
        raise ValueError("All sequences must have the same dtype.")

    seq_len = max([s.shape[0] for s in sequences])
    seq_shape = tuple(sequences[0].shape[1:])
    seq_dtype = sequences[0].dtype
    padded_sequences = torch.ones(len(sequences), seq_len, *seq_shape, dtype=seq_dtype) * pad_value

    for i,s in enumerate(sequences):
        padded_sequences[i, -s.shape[0]:, ...] = s

    return padded_sequences

def load_sequential_data(features_dir, layer, fmri_dir, subject, ROI):
    """ Load sequence data for using with a N-CDE model.
        Also, adds time dimension to sequences.

    Parameters
    ----------
    features_dir : str
        path to features
    subject : int or str
        layer number if int, 'layer_#num' if str
    fmri_dir : str
        path to fMRI data
    subject : int or str
        subject number if int, 'sub#num' if str
    ROI : str
        name of ROI.

    Returns
    -------
    X : np.array
        a tensor of sequence of activations of shape (#videos, #sequence_len, #features)
    Y : np.array
        a tensor of fMRI responses of shape (#videos, #voxels)
    X_test : np.array
        a tensor of test activations of shape (#videos, #sequence_len, #features)
    """
    layer_str = f"layer_{layer}" if type(layer) == int else layer
    activation_files = glob.glob(features_dir + "/*" + layer_str + ".npy")
    activation_files.sort()

    def __load_to_torch(file_path):
        x = torch.from_numpy(np.concatenate(np.load(file_path)))
        x = x.flatten(1) # have to flatten because of adding time channel (singel number)
        return torch.cat([x, torch.arange(1, x.shape[0]+1).unsqueeze(1)], dim=1) 

    X = __pad_sequence([__load_to_torch(acf) for acf in activation_files])
    X_train = X[0:1000]
    Y_train = torch.from_numpy(load_fmri(fmri_dir, subject, ROI).mean(axis=1).astype(np.float32))
    X_test = X[1000:]

    return X_train, Y_train, X_test

X_train, Y_train, X_test = load_sequential_data("./data/alexnet_frames/", "layer_5", "./data/participants_data_v2021/", 1, "V1")
