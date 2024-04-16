from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import csv

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    This line creates an ArgumentParser object from the argparse standard library. 
    This class is used to parse *command line arguments* in Python.
    """
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    """
    This line prints the parsed arguments so the user can see what values have been 
    passed to the script from the command line.
    """
    print ("User-entered parameters to train: ", opt)
    
    #Pendiente solucionar
    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available GPU: ", device)

    # Create directories if don't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    """
    Get data configuration throught "opt" object of Namespace type.
    Adds other atributes to the object opt, in addition tho those entered by the user.
    In this case, the paths to the configuration files.
    The Namespace type is a data type provided by Python's argparse module. 
    Represents a simple and useful namespace for storing command line arguments 
    and their associated values. Namespace is simply a data structure that 
    allows the values of command line arguments to be accessed in an 
    organized and easy-to-use way in a Python script.
    """
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"]) #Here, uses the function load_clases(path) in utils

    # Initiate model
    model = Darknet(opt.model_def).to(device) # opt.model_def contains the path to the model definition file (.cfg format). Then move the model to device (GPU, CPU)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint - Load pretrained weights through command line
    if opt.pretrained_weights:
        # This line checks if the pre-trained weights file has a .pth extension
        if opt.pretrained_weights.endswith(".pth"): # .pth files are designed specifically to work with PyTorch, meaning they are easy to load and manage using the functions and methods provided by PyTorch.
            model.load_state_dict(torch.load(opt.pretrained_weights)) #load the weights into the model using PyTorch's load_state_dict() method.
        else:
            model.load_darknet_weights(opt.pretrained_weights) # If the pre-trained weights are not in .pth format, this suggests that they are in Darknet format. In this case, the load_darknet_weights() method defined in the Darknet class is used to load the weights into the model.
    
    # Aquí podríamos considerar usar otro archivo pth, más actualizado para el proyecto.

    # Get dataloader - create a DataLoader in PyTorch to load the data from the training dataset
    """
    Here the ListDataset dataset is instantiated using the path to the training directory train_path. 
    The argument augment=True indicates that data augmentation will be performed during training 
    (e.g. rotations, cropping, etc.), and multiscale=opt.multiscale_training indicates whether 
    training at multiple scales will be allowed. This creates a dataset object that contains the 
    training data along with the specified transformations.
    """
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    """
    Here the DataLoader is created using the dataset data set created earlier. 
    Various arguments are specified such as batch_size, shuffle...
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size, 
        shuffle=True, # This indicates that the data is shuffled before batching, which helps improve the generalization of the model.
        num_workers=opt.n_cpu, # This specifies the number of threads to use to load the data efficiently. More threads can make data loading faster, but CPU usage will also increase.
        pin_memory=True, # This is set to true to speed up data transfer to the GPU if it is being used, by preventing data from being copied to main memory.
        collate_fn=dataset.collate_fn, # This specifies a custom grouping function that is used to combine the data in batches. In this case, collate_fn defined in the dataset is used.
    )
    """
    Note: 
    If we have a GPU available as an accelerator in model training, the num_workers=opt.n_cpu parameter 
    in the DataLoader will control the number of threads used to load the data. Although the parameter 
    refers to "CPU" (n_cpu), it is still relevant even if the GPU is used. The reason is that although 
    the GPU is responsible for the calculation of parallel operations, the CPU is still responsible 
    for other tasks, such as loading data and preparing batches. Additional threads (num_workers) 
    help speed up data loading by allowing multiple data loading operations to be performed in parallel.
    It is common to set num_workers to a value greater than zero to take advantage of CPU parallelism 
    and speed up data loading. However, a very high number of threads can overload the CPU and have a 
    negative impact on overall performance.
    """
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01) # I changed it
    metrics = [
        "grid_size", # The size of the grid used to divide the input image.
        "loss", # This loss generally includes terms of coordinate loss, confidence loss, and classification loss.
        "x",
        "y",
        "w",
        "h", # x, y, w, h: These metrics represent the coordinate loss for the coordinates x, y (position of the object on the grid) and w, h (width and height of the object).
        "conf", # Measures the accuracy of object detection in terms of the confidence assigned to the detections.
        "cls", # Measures the classification accuracy of detected object classes.
        "cls_acc", # Classification accuracy of detected object classes
        "recall50", # The recall rate of object detections with a confidence threshold of 50% and 75%, respectively.
        "recall75",
        "precision", # The accuracy of object detections
        "conf_obj", # The loss of trust for present objects.
        "conf_noobj", # The loss of trust for absent objects.
    ]
    
    #Recall50 and Recall75: They are the recall rate (the proportion of positive examples 
    #that were correctly identified) calculated from the confidence thresholds of 50% and 75% 
    #respectively. For example, recall50 of 0.90 means that 90% of the positive examples were 
    #correctly identified with a 50% confidence threshold.

    #conf_obj and conf_noobj: They are object and non-object confidence measures respectively. 
    #Conf_obj is the average confidence assigned to correctly detected objects, 
    #while Conf_noobj is the average confidence assigned to places where no object was detected. 
    #A high value in Conf_obj indicates that the model is confident in its object detections, 
    #while a low value in Conf_noobj suggests that the model is not generating false positives 
    #in areas where there are no objects.

    #cls_acc: It is the classification accuracy, that is, the proportion of classes correctly 
    #predicted in all detections. It is calculated as the percentage of correctly predicted 
    #classes among all detections.

    #Precision: It is a measure of the precision of object detections, that is, the proportion 
    #of positive detections that are truly positive. It is calculated as the number of true 
    #positives divided by the total number of detections (true positives plus false positives). 
    #The precision is calculated for each class individually and then averaged.

    average_metrics = []

    for epoch in range(opt.epochs):
        model.train() # Sets the model in training mode
        start_time = time.time() # Mark the begig of train time
        """
        The loop iterates over batches of data provided by the DataLoader. 
        Each batch consists of images (imgs) and their corresponding targets (targets), 
        which contain the labels of the objects in those images.
        """
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            
            """
            Batches_done: Represents the total number of batches processed from the start of training 
            to the current batch at the current time. It is a way to keep track of training 
            progress and provide feedback on how many batches have been processed in total 
            in the current training.
            """
            batches_done = len(dataloader)* epoch + batch_i

            #In this line, the input images (imgs) and targets are sent to the specified computing device (device)
            """
            In recent versions of PyTorch (1.0 and later), the use of Variable is no longer 
            necessary as Tensors have the ability to track gradient by default. So here we've changed the code
            """
            #imgs = Variable(imgs.to(device))
            #targets = Variable(targets.to(device), requires_grad=False) #requires_grad=False: It's not neccesary to calculate gradients of labels (targets)

            #Send images to the device and enable gradient tracking
            imgs = imgs.to(device).requires_grad_()
            #Send targets to the device and disable gradient tracking
            targets = targets.to(device)
            
            """
            Here the forward propagation (forward pass) of the data through the model is performed. 
            The input tensors (imgs and targets) are passed to the model, which then performs the 
            necessary operations (such as convolutions, activations, etc.) to generate predictions 
            (outputs) and calculate the loss. The loss represents the difference between the model 
            predictions and the actual labels and is a measure of how well the model is performing 
            on the training data.
            """
            loss, outputs = model(imgs, targets)
            """
            After calculating the loss, this line performs gradient backpropagation (backward pass) 
            through the computation graph that PyTorch has constructed during forward propagation. 
            During backpropagation, PyTorch calculates loss gradients with respect to model parameters, 
            allowing model weights to be adjusted to minimize loss in future training iterations. 
            That is, this line calculates the gradients of the loss with respect to the model parameters,
            allowing model optimization through gradient descent.
            """
            loss.backward()

            """
            This technique of accumulating gradients before updating weights can be useful in situations
            where GPU memory is limited and an entire large batch cannot be processed at once. 
            Accumulating gradients over multiple iterations can help mitigate this issue and enable 
            effective training of resource-constrained models.
            """
            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            #Training progress in each iteration of the training loop
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            #This line creates a list that contains another nested list. 
            #The outer list will contain all the rows of the metrics table, 
            #and the inner list will represent the first row of table headers.
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                #tensorboard_log = []
                #for j, yolo in enumerate(model.yolo_layers):
                #    for name, metric in yolo.metrics.items():
                #        if name != "grid_size":
                #            tensorboard_log += [(f"{name}_{j+1}", metric)]
                #tensorboard_log += [("loss", loss.item())]
                #logger.list_of_scalars_summary(tensorboard_log, batches_done)

                for metric in metrics:
                    if metric == "grid_size":
                        continue  # No incluir "grid_size" en los cálculos
                    metric_values = []
                for yolo in model.yolo_layers:
                    metric_value = yolo.metrics.get(metric, 0)
                    metric_values.append(metric_value)
                metric_average = sum(metric_values) / len(metric_values)
                average_metrics.append(metric_average)

                csv_file = "average_metrics.csv"
                with open(csv_file, "w", newline="") as csvfile:
                    fieldnames = ["Metric", "Average Value"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for i, metric in enumerate(metrics):
                        if metric == "grid_size":
                            continue
                        writer.writerow({"Metric": metric, "Average Value": average_metrics[i]})


            #log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            #print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"yolov3_ckpt_%d.pth" % epoch)

    end_time = time.time()
    training_duration = (end_time - start_time)/60
    print(f"Train time: {training_duration:.2f} minuts")