#!/usr/bin/env python
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import pickle
import optuna

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

import util
from gnn import GNN

def objective_with_dataset(dataset):

    def objective(trial):
        num_layers = trial.suggest_int('num_layers', 1, 10)
        num_mlp_layers = trial.suggest_int('num_mlp_layers', 1, 10)
        hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
        final_dropout = trial.suggest_uniform('final_dropout', 0, 0.5)
        graph_pooling_type = trial.suggest_categorical('graph_pooling_type', ['max', 'average', 'sum'])
        neighbor_pooling_type = trial.suggest_categorical('neighbor_pooling_type', ['max', 'average', 'sum'])
        batchsize = trial.suggest_int('batchsize', 16, 128)

        device = chainer.get_device(-1)
        # Classification
        model = L.Classifier(GNN(num_layers, num_mlp_layers, dataset.graphs[0].node_features.shape[1],
                                    hidden_dim, dataset.graphs[0].node_features.shape[1], final_dropout, graph_pooling_type, neighbor_pooling_type, "Regression"))

        # choose the using device
        model.to_device(device)
        device.use()

        # Setup an optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        # split the dataset into traindata and testdata
        train, test = chainer.datasets.split_dataset_random(dataset, int(dataset.__len__() * 0.9))
        train_iter = chainer.iterators.SerialIterator(train, batchsize)
        test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

        # Set up a trainer
        updater = training.updaters.StandardUpdater(train_iter, optimizer, device=device, converter=dataset.converter)
        trainer = training.Trainer(updater, (300, 'epoch'), out= "result/hypara")

        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(test_iter, model, device=device, converter=dataset.converter))

        trainer.extend(extensions.LogReport(filename='log_{}.dat'.format(trial.number)))
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss_{}.png'.format(trial.number)))

        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy_{}.png'.format(trial.number)))

        # Run the training
        trainer.run()

        # save the model
        chainer.serializers.save_npz('./result/hypara/{0}.model'.format(trial.number), model)

        # return the AUC
        graphs, target = dataset.converter(test, device)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y_pred = model.predictor(graphs)
        y_pred.to_cpu()
        y_pred = y_pred.array
        target = chainer.cuda.to_cpu(target)

        return 1 - sklearn.metrics.roc_auc_score(target, y_pred[:,1])

    return objective


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The simple implementation of GIN using sparse matrix multiplication')
    parser.add_argument('--dataset', type=str, default="mixed",
                        help='name of dataset (default: mixed)')
    parser.add_argument('--degree_as_tag', type=str, default="binary",
                		help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--dataset_num', type=int, default=2000, help='# of dataset')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load Grapdata
    device = chainer.get_device(-1)
    dataset = util.GraphData(args.dataset, args.degree_as_tag, "Task1", args.dataset_num, device)

    study_name = 'binary-study'
    study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db', load_if_exists=True)
    study.optimize(objective_with_dataset(dataset), n_trials=100)

    print(study.best_params, study.best_value)
    hist_df = study.trials_dataframe()
    hist_df.to_csv("binary_hypara_search.csv")
