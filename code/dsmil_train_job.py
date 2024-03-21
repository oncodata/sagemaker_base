import torch
import os
import boto3
import argparse
import resource
from utils.datasets import WSIData
from torch.utils.data import DataLoader
from utils.test_dsmil import test_dsmil_model
from utils.train_dsmil import train_dsmil_model
from models.dsmil import MILNet, FCLayer, BClassifier


# Sets the current resource limit on the number of file descriptors that the current process can open
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# This code is used to disable the profiling executor and profiling mode in PyTorch's JIT (Just-In-Time) compiler.
if (torch._C, '_jit_set_profiling_executor'):
    torch._C._jit_set_profiling_executor(False)
if (torch._C, '_jit_set_profiling_mode'):
    torch._C._jit_set_profiling_mode(False)


def parse():
    parser = argparse.ArgumentParser(description='Load model configuration')
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    parser.add_argument(
        "--s3-manual-save-path",
        type=str,
        default="",
        help="Caminho para salvar dados manualmente no S3",
    )

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--config', type=str, default=None, help='Configuration file to apply on top of base')
    return parser.parse_args()


def main():
    # Check if smdistributed is available
    try:
        import smdistributed.dataparallel.torch.distributed as dist
        smdistributed_available = True
    except ImportError:
        import torch.distributed as dist
        smdistributed_available = False

    args = parse()
    if smdistributed_available:
        dist.init_process_group()
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    else:
        # Initialize PyTorch distributed without smdistributed
        import tempfile
        import os

        file_path = os.path.join(tempfile.gettempdir(), "shared_file")
        backend = 'nccl'
        dist.init_process_group(backend=backend, init_method='file://%s' % file_path, world_size=1, rank=0)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.local_rank = int(os.getenv("LOCAL_RANK", 0))

    s3_manual_save_path = args.s3_manual_save_path
    save_path = args.model_dir

    torch.manual_seed(args.seed)

    # Inserir código a ser executado abaixo ------------------------------------------------------------>

    FEATURES_FOLDER = args.train
    FEATS_SIZE = 256
    NUM_CLASSES = 1
    TASK = 'stomach_urg'
    lr = 5e-3
    weight_decay = 1e-4
    CLASSES = ['Primary Tumor', 'Solid Tissue Normal']
    OVERSAMPLE_MINORITY = True
    STAGES_FILES = {'train': 'stomach_json_train.json'}  # {'train': 'prostate_json_train.json', 'val': 'prostate_json_test.json'}
    USE_VALIDATION = len(STAGES_FILES.keys()) > 1
    TEST_FILE = 'stomach_json_test.json'

    def collate_fn_multiple_size(data):
        feats, labels, lengths = zip(*data)
        return feats, torch.LongTensor(labels), lengths

    loader = {}
    for stage in STAGES_FILES.keys():
        dataset = WSIData(features_dir=FEATURES_FOLDER, stage=stage, task=TASK, json_name=STAGES_FILES[stage],
                          classes_to_use=CLASSES, oversample_minority=OVERSAMPLE_MINORITY)
        # Using DistributedSampler ensures that each training instance in your SMDDP cluster processes a unique subset of the data during each training step.
        # This helps improve training efficiency and convergence by reducing redundant computation and ensuring that each instance sees a diverse set of training examples.
        if backend == 'smddp':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, collate_fn=collate_fn_multiple_size, sampler=sampler)
        else:
            data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, collate_fn=collate_fn_multiple_size)
        loader.update({stage: data_loader})

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cpu")
    i_classifier = FCLayer(in_size=FEATS_SIZE, out_size=NUM_CLASSES).to(device)
    b_classifier = BClassifier(input_size=FEATS_SIZE, output_class=NUM_CLASSES, dropout_v=0., nonlinear=1).to(device)
    if smdistributed_available:
        model = dist.new_model(MILNet(i_classifier, b_classifier).to(device))
    else:
        model = MILNet(i_classifier, b_classifier).to(device)

    train_dsmil_model(model, loader, num_runs=1, lr=lr, weight_decay=weight_decay,
                      num_epochs=args.epochs, num_classes=NUM_CLASSES,
                      feats_size=FEATS_SIZE, use_validation=USE_VALIDATION)

    test_dataset = WSIData(features_dir=FEATURES_FOLDER, stage='test', task=TASK,
                           json_name=TEST_FILE, classes_to_use=CLASSES,
                           oversample_minority=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=True)
    auc_value, test_labels, test_predictions = test_dsmil_model(model, test_loader,
                                                                feats_size=256, num_classes=1)

    print(auc_value)

    # When the training job finishes, the container and its file system will be
    # deleted, with the exception of the /opt/ml/model and /opt/ml/output directories.
    # Use /opt/ml/model to save the model checkpoints. These checkpoints will be uploaded to the default S3 bucket.
    if args.rank == 0:
        model_name = 'dsmil.pth'
        save_model_path = os.path.join(save_path, model_name)
        torch.save(model.state_dict(), save_model_path)
        # Save the model in s3_manual_save_pat
        if s3_manual_save_path:
            s3 = boto3.client('s3')
            bucket_name = s3_manual_save_path.split('/')[0]
            object_key = s3_manual_save_path.split(bucket_name)[1] + '/' + model_name
            s3.upload_file(save_model_path, bucket_name, object_key)


if __name__ == '__main__':
    main()