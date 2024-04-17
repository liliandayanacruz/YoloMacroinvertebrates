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

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
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
    
    print ("User-entered parameters to train: ", opt)
    
    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available GPU: ", device)

    # Create directories if don't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"]) #Here, uses the function load_clases(path) in utils

    # Initiate model
    model = Darknet(opt.model_def).to(device) 
    model.apply(weights_init_normal)

    # If specified we start from checkpoint - Load pretrained weights through command line
    if opt.pretrained_weights:
        # This line checks if the pre-trained weights file has a .pth extension
        if opt.pretrained_weights.endswith(".pth"): # .pth files are designed specifically to work with PyTorch.
            model.load_state_dict(torch.load(opt.pretrained_weights)) #load the weights into the model.
        else:
            model.load_darknet_weights(opt.pretrained_weights) # If the pre-trained weights are not in .pth 
    
    # Get dataloader - create a DataLoader in PyTorch to load the data from the training dataset
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size, 
        shuffle=True, # The data is shuffled before batching, which helps improve the generalization of the model.
        num_workers=opt.n_cpu, # This specifies the number of threads to use to load the data efficiently. 
        pin_memory=True, # This is set to true to speed up data transfer to the GPU if it is being used
        collate_fn=dataset.collate_fn, # Custom grouping function that is used to combine the data in batches. 
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01) # I changed it
    metrics = [
        "grid_size", # The size of the grid used to divide the input image.
        "loss", # This loss generally includes terms of coordinate loss, confidence loss, and classification loss.
        "x",
        "y",
        "w",
        "h", # x, y, w, h: Coordinates x, y (position of the object on the grid) and w, h (width and height).
        "conf", # Measures the accuracy of object detection in terms of the confidence assigned to the detections.
        "cls", # Measures the classification accuracy of detected object classes.
        "cls_acc", # Classification accuracy of detected object classes
        "recall50", # Detections with a confidence threshold of 50% and 75%, respectively.
        "recall75",
        "precision", # The accuracy of object detections
        "conf_obj", # The loss of trust for present objects.
        "conf_noobj", # The loss of trust for absent objects.
    ]
   
    for epoch in range(opt.epochs):
        model.train() # Sets the model in training mode
        start_time = time.time() # Mark the begig of train time. 
        #The loop iterates over batches of data provided by the DataLoader. Each batch has images and targets
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            #Represents the total number of batches processed from the start of training 
            #to the current batch at the current time. 
            batches_done = len(dataloader)* epoch + batch_i
            
            #Send images and targets to the device and enable gradient tracking
            imgs = imgs.to(device).requires_grad_()
            targets = targets.to(device)
            
            #Forward propagation (forward pass)
            loss, outputs = model(imgs, targets)
            # Performs gradient backpropagation (backward pass)
            loss.backward()

            # Accumulating gradients before updating weights 
            if batches_done % opt.gradient_accumulations:
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            # A list that contains another nested list. 
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)
                
            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

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