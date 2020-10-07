import datetime as dt
import numpy as np
import os
import shutil
import time
import torch
import visdom

from abc import ABC, abstractmethod
from termcolor import colored
from utils import torch as utor
from utils import train as utr

class NetworkTrainer(ABC):

    def __init__(self, arguments):
        self.args = arguments
        self.data_loader_train = None
        self.data_loader_validation = None

        self.epoch_current = 0
        self.epoch_start = 0

        self.iteration_current = 0
        self.lr = 0

        self.dict_paths = {}
        self.dict_meter_loss_train = {}
        self.dict_meter_loss_val = {}

        self.dict_curves_loss_train = {}
        self.dict_curves_loss_validation = {}
        self.dict_curves_loss_train_per_validation = {}

        self.device = None

        self.loss_net = None
        self.logger = None
        self.optimizer = None

        self.visualizer = None

    @abstractmethod
    def load_from_checkpoint(self):
        pass

    def __init_dataloaders(self):

        time_init_ds_train_start = time.perf_counter()
        self.data_loader_train = self.init_dataloader_train()

        time_init_ds_train = time.perf_counter() - time_init_ds_train_start

        self.logger.info("Time to init train dataset: {0} Num of Samples: {1}".format(time_init_ds_train,
                                                                                      len(self.data_loader_train) * self.args.size_batch))

        time_init_ds_validation_start = time.perf_counter()
        self.data_loader_validation = self.init_dataloader_validation()
        time_init_ds_validation = time.perf_counter() - time_init_ds_validation_start

        self.logger.info("Time to init validation dataset: {0} Samples: {1}".format(time_init_ds_validation,
                                                                                    len(self.data_loader_validation) * self.args.size_batch))

        return True

    @abstractmethod
    def init_dataloader_train(self):
        pass

    @abstractmethod
    def init_dataloader_validation(self):
        pass

    @abstractmethod
    def init_output_dirs(self):
        pass

    def init_logger(self):
        self.logger = utr.setup_logger(self.args.name_train, self.args.path_log, "log.txt")
        self.logger.info("%s", repr(self.args))

        return True

    @abstractmethod
    def init_network(self):
        pass

    @abstractmethod
    def init_losses(self):
        pass

    @abstractmethod
    def init_optimizer(self):
        pass

    def init_visualizer(self):
        self.visualizer_save_folder = os.path.join(self.args.path_log, "visdom")
        
        self.visualizer = visdom.Visdom(port=self.args.port_vis,
                                            env=self.args.name_train)
        
        if not os.path.exists(self.visualizer_save_folder) and not os.path.isdir(self.visualizer_save_folder):
            os.mkdir(self.visualizer_save_folder)

        return True

    def init_train_device(self):

        if (self.args.use_gpu >= 1):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.id_gpu)

            self.device = utor.get_gpu_device(0)
            torch.backends.cudnn.benchmark = True
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
            print(colored("WARNING: Running on CPU only", 'white', 'on_red'))

            self.device = utor.get_cpu_device()

        return True

    @abstractmethod
    def move_network_to_device(self):
        pass

    @abstractmethod
    def put_network_in_train(self):
        pass

    @abstractmethod
    def put_network_in_eval(self):
        pass

    @abstractmethod
    def reset_meters_train(self):
        for meter in self.dict_meter_loss_train.values():
            meter.reset()

    @abstractmethod
    def reset_meters_validation(self):
        for meter in self.dict_meter_loss_val.values():
            meter.reset()

    @abstractmethod
    def do_step_fwd(self, data):
        pass

    @abstractmethod
    def compute_loss(self, results_forward_step):
        pass

    def __init_meters(self):
        self.init_meters_train()
        self.init_meters_validation()

        return True

    @abstractmethod
    def init_meters_train(self):
        pass

    @abstractmethod
    def init_meters_validation(self):
        pass

    @abstractmethod
    def init_meters_validation(self):
        pass

    def __init_curves(self):
        self.init_curves_train()
        self.init_curves_validation()

        return True

    @abstractmethod
    def init_curves_train(self):
        pass

    @abstractmethod
    def init_curves_validation(self):
        pass

    def prepare(self):

        if not self.init_output_dirs():
            raise Exception('init_output_dirs() failed.')

        if not self.init_logger():
            raise Exception('init_logger() failed.')

        if not self.init_visualizer():
            raise Exception('init_visualizer() failed.')

        if not self.__init_curves():
            raise Exception('init_curves() failed.')

        if not self.__init_meters():
            raise Exception('init_meters() failed.')

        if not self.init_train_device():
            raise Exception('init_train_device() failed.')

        if not self.__init_dataloaders():
            raise Exception('init_dataloader() failed.')

        if not self.init_network():
            raise Exception('init_network() failed.')

        if not self.init_losses():
            raise Exception('init_losses() failed.')

        if not self.init_optimizer():
            raise Exception('init_optimizer() failed.')

    @abstractmethod
    def decay_learning_rate(self):
        pass

    def get_console_string_train(self, date, index_batch, epoch, iteration, len_dataset, learning_rate, res_losses, time_fwd):
        return utr.get_console_string(date=date,
                                      iteration=iteration,
                                      epoch=epoch,
                                      index_batch=index_batch,
                                      len_loader=len_dataset,
                                      learning_rate=learning_rate,
                                      loss_net=res_losses,
                                      time_fwd=time_fwd)

    def get_console_string_validation(self, date, index_batch, len_dataset, res_losses):

        return utr.get_console_string(date=date,
                                      iteration=0.0,
                                      epoch=0,
                                      index_batch=index_batch,
                                      len_loader=len_dataset,
                                      learning_rate=0.0,
                                      loss_net=res_losses,
                                      time_fwd=0.0)

    @abstractmethod
    def update_meters_train(self, res_losses_train, batch_losses_train):
        pass

    @abstractmethod
    def update_meters_validation(self, res_losses_validation, batch_losses_validation):
        pass

    def update_curves_train(self, dict_avg_losses_train):
        """
        Given the moving average from the loss meter update the dictionary of curves
        :param dict_avg_losses_train: the moving average for the current interation for each loss
        :return: nothing
        """

        for name, curve in dict_avg_losses_train.items():
            self.dict_curves_loss_train[name].append(dict_avg_losses_train[name])

    def update_curves_validation(self, dict_avg_losses_validation):
        """
        Update curves for loss visdom visualizer. Append a new entry into the dictionary of validation losses and train losses for validation
        :param dict_avg_losses_validation: moving average for the validation loss
        :return: nothing
        """

        for name in dict_avg_losses_validation:
            self.dict_curves_loss_train_per_validation[name].append(self.dict_meter_loss_train[name].value()[0])
            self.dict_curves_loss_validation[name].append(dict_avg_losses_validation[name])

    @abstractmethod
    def update_visualizer_train(self, res_fwd, dict_avg_train_losses, dict_batch_train_losses):
        pass

    @abstractmethod
    def update_visualizer_validation(self, dict_curve_validation_losses, dict_curve_train_per_validation_losses):

        for t_train, t_val in zip(dict_curve_train_per_validation_losses.items(), dict_curve_validation_losses.items()):
            name_win = 'loss val ' + t_train[0]

            ranges_validation = np.column_stack((np.arange(len(t_train[1])), np.arange(len(t_val[1]))))
            values_vaidation = np.column_stack((np.array(t_train[1]), np.array(t_val[1])))

            self.visualizer.line(X=ranges_validation, Y=values_vaidation, win=name_win,
                                 opts=dict(title=name_win, legend=["train", "validation"],markersize=2, ), )

    @abstractmethod
    def update_visualizer_validation_per_iteration(self, res_fwd, dict_batch_validation):
        pass

    def save_visualizer_env(self):
        """
        Saves all the plots in the environment of the running training in the log folder.
        The name of the saved file is name_train.json
        """
        self.visualizer.save([self.args.name_train])
        shutil.copy2(os.path.join(os.path.expanduser('~'), ".visdom", self.args.name_train+".json"), self.visualizer_save_folder)

    @abstractmethod
    def compute_metric_validation(self, res_step_fwd_validation):
        pass

    def do_epoch_validation(self):
        dict_avg_losses = {}

        for it, data in enumerate(self.data_loader_validation):

            res_step_fwd_validation = self.do_step_fwd(data)
            avg_losses_validation, batch_losses_validation = self.compute_metric_validation(res_fwd_validation=res_step_fwd_validation)

            # Update meters
            dict_avg_losses = self.update_meters_validation(avg_losses_validation, batch_losses_validation)

            # Print string
            str_console_validation = self.get_console_string_validation(date=dt.datetime.now(),
                                                                        index_batch=it,
                                                                        len_dataset=len(self.data_loader_validation),
                                                                        res_losses=dict_avg_losses)
            print(colored(str_console_validation, 'white', 'on_blue'))

            if it % self.args.step_per_viz == 0:

                self.update_visualizer_validation_per_iteration(res_fwd=res_step_fwd_validation,
                                                                dict_batch_validation_losses=batch_losses_validation)

        return dict_avg_losses

    def start_train(self):
        # Load from checkpoint
        self.load_from_checkpoint()

        # Move to device
        self.move_network_to_device()

        for self.epoch_current in range(self.epoch_start, self.args.epochs_max):

            self.reset_meters_train()
            self.put_network_in_train()

            if self.epoch_current is not self.epoch_start:
                self.decay_learning_rate()

            for it, data in enumerate(self.data_loader_train):
                self.optimizer.zero_grad()

                time_before_step = time.perf_counter()

                res_fwd_step = self.do_step_fwd(data)

                time_fwd = time.perf_counter() - time_before_step

                res_losses, batch_res_losses = self.compute_loss(res_fwd_step)
                self.loss_net.backward()

                self.optimizer.step()
                dict_avg_losses = self.update_meters_train(res_losses, batch_res_losses)

                str_console_train = self.get_console_string_train(date=dt.datetime.now(),
                                                                  index_batch=it,
                                                                  epoch=self.epoch_current,
                                                                  iteration=self.iteration_current,
                                                                  len_dataset=len(self.data_loader_train),
                                                                  learning_rate=self.lr,
                                                                  res_losses=dict_avg_losses,
                                                                  time_fwd=time_fwd)
                print(str_console_train)

                if self.iteration_current % self.args.step_per_viz == 0:

                    # Save results in curves
                    self.update_curves_train(dict_avg_losses_train=dict_avg_losses)

                    # Update visualizer
                    self.update_visualizer_train(res_fwd=res_fwd_step,
                                                 dict_avg_train_losses=dict_avg_losses,
                                                 dict_batch_train_losses=batch_res_losses)
                    self.logger.info(str_console_train)

                if self.iteration_current % self.args.step_per_save == 0:

                    self.reset_meters_validation()

                    with torch.no_grad():
                        self.put_network_in_eval()

                        time_init_validation = time.perf_counter()

                        dict_avg_losses_validation = self.do_epoch_validation()

                        # Update curvers validation
                        self.update_curves_validation(dict_avg_losses_validation)

                        self.update_visualizer_validation(dict_curve_validation_losses=self.dict_curves_loss_validation,
                                                          dict_curve_train_per_validation_losses=self.dict_curves_loss_train_per_validation)

                        time_validation = time.perf_counter() - time_init_validation

                        self.logger.info("Time Validation: {}".format(time_validation))

                        self.save_checkpoint()

                    print(colored('Network saved at %s' % self.dict_paths['checkpoint'], 'white', 'on_red'))

                self.iteration_current += 1

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass

