from model import *
from data_loader import *
import argparse
import json
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d' , type=str, default='data/', help='Path to dataset') # dataset path
    parser.add_argument('--config', '-c' , type=str, default='config.json', help='Path to config file') # training config
    parser.add_argument('--model', '-m' , type=str, default=None, help='Path to model file', required=False) # if resume training
    parser.add_argument('--device', '-g' , type=str, default='cpu', help='Device to use for training') # device to use for training
    return parser.parse_args()


def parse_config_json(config_json_file):
    with open(config_json_file, 'r') as f:
        config = json.load(f)
    
    return config


def e2e_pipeline(args):
    
    print(f'##################\n# Pipeline start #\n##################')
    
    logging.info("Parsing config file...")
    config = parse_config_json(args.config)
    
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    LEARNING_RATE = config['learning_rate']
    OPTIMIZER = config['optimizer']
    LOSS_FUNCTION = config['loss_function']
    TRAIN_VAL_TEST_SPLIT = config['train_val_test_split']
    MODEL_PARAMETERS = config['model_parameters']
    
    logging.info("Loading data...")
    dataset_as_dict = load_data(args.dataset)
    
    logging.info("Splitting dataset...")
    train, test, val = split_dataset(dataset_as_dict, 
                                     split_ratio=TRAIN_VAL_TEST_SPLIT, 
                                     validation=True)
    
    logging.info("Converting to torch.datasets...")
    train_dataset = make_dataset(train)
    test_dataset = make_dataset(test)
    val_dataset = make_dataset(val)
    
    logging.info("Making dataloaders...")
    train_dataloader, test_dataloader, val_dataloader = make_dataloaders(batch_size=BATCH_SIZE, 
                                                                         train=train_dataset, 
                                                                         test=test_dataset, 
                                                                         val=val_dataset)

    logging.info("Loading model...")    
    model = get_model(parameters=MODEL_PARAMETERS, 
                      resume=args.model)
    
    criterion = get_criterion(loss_fn=LOSS_FUNCTION)
    
    optimizer = get_optimizer(optimizer=OPTIMIZER, 
                              model=model, 
                              lr=LEARNING_RATE)
    
    logging.info("Training model...")
    model, model_metrics = train_model(model=model, 
                                       train_dataloader=train_dataloader, 
                                       val_dataloader=val_dataloader, 
                                       optimizer=optimizer, 
                                       loss_fn=criterion, 
                                       epochs=EPOCHS)
    logging.info("Model trained.")
    
    logging.info("Testing model...")
    test_accuracy, test_loss = test_model(model=model, 
                                            test_dataloader=test_dataloader, 
                                            loss_fn=criterion)
    
    print(f'###################\n# Pipeline end #\n###################')

    print(f'\nBEST Train accuracy:\t{max(model_metrics["train_accuracy"]):.4f}')
    print(f'BEST Train loss:\t{min(model_metrics["train_loss"]):.4f}')
    print(f'Test accuracy:\t{test_accuracy:.4f}')
    print(f'Test loss:\t{test_loss:.4f}')

if __name__ == '__main__':
    args = parse_args()
    e2e_pipeline(args)