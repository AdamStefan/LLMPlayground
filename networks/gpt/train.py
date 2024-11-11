import numpy as np
import os
import torch


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch._inductor.config as config
import torch.distributed as dist
from contextlib import nullcontext
from dataclasses import dataclass
import tiktoken
from networks.gpt.gpt_model import GPTModel, Config
from data.data_utils import DistributedDataLoader
from networks.gpt.gpt_helpers import load_from_hf_weights
import inspect
import math
import time
from loggers.tensorboard import TensorBoardLogger
import yaml
from torch.optim import Optimizer


@dataclass
class TrainParameters:
    experiment_name: str
    train_dataset_path_bin: str
    val_dataset_path_bin: str
    model: str
    output_dir: str
    batch_size: int = 4
    total_batch_size: int = 256
    sequence_length: int = 64
    number_of_iterations: int = 10000
    inference_only: bool = False
    learning_rate: float = 1e-4
    warmup_iters: int = 0
    learning_rate_decay_frac: float = 1.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    val_loss_every: int = 1000
    val_max_steps: int = 20
    sample_every: int = 0
    overfit_single_batch: bool = False
    tensorcores: bool = True
    device: str = ""
    compile: bool = False
    flash: bool = True
    # dtype: str = "float32"
    dtype: str = "bfloat16"
    zero_stage: int = 0


def configure_optimizers(model: GPTModel, weight_decay, learning_rate, betas, device_type, zero_stage) -> Optimizer:
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [{"params": decay_params, "weight_decay": weight_decay}, {"params": nodecay_params, "weight_decay": 0.0}]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    print0(f"using fused AdamW: {use_fused}")
    if zero_stage == 1:
        print0("using ZeroRedundancyOptimizer")
        optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW, lr=learning_rate, betas=betas, fused=use_fused)
        optimizer.add_param_group(optim_groups[1])
    else:
        print0("using regular AdamW")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
    return optimizer


def _save_checkpoint(output_dir: str, model: GPTModel, optimizer: Optimizer, step: int, model_args: TrainParameters, best_validation_loss: float) -> None:
    """
    Called when saving a model checkpoint, use to persist state.

    Args:
        experiment: the current instance.
    """
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # remember DDP module to continue training
        model = model.model
    checkpoint = {"state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(), "train_args": model_args.__dict__, "step": step, "best_validation_loss": best_validation_loss}
    torch.save(checkpoint, os.path.join(output_dir, "ckpt.pt"))


def _load_checkpoint(output_dir: str, model: GPTModel, device: torch.device) -> dict:
    """Called when loading a model checkpoint, use to reload state.

    Args:
        experiment: the current instance.
    """
    checkpoint = torch.load(os.path.join(output_dir, "ckpt.pt"), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def train(args: TrainParameters):
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        # select the device
        if args.device:
            # provided explicitly by the user
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = "cuda" if "cuda" in device else "cpu"

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    B, T = args.batch_size, args.sequence_length
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision("high")

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # create the logging directory if it does not exist
    logfile = None
    output_dir = args.output_dir
    if args.experiment_name is not None:
        output_dir = os.path.join(output_dir, args.experiment_name)
    logger = None

    if output_dir:
        logger = TensorBoardLogger(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logfile = os.path.join(output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

        hparams_file = os.path.join(output_dir, "hparams.yaml")
        with open(hparams_file, "w", newline="") as fp:
            yaml.dump(args.__dict__, fp)

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    if args.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": Config(block_size=1024, vocab_size=50257, n_layers=12, n_heads=12, dim=768, use_flash=args.flash),
            "d24": Config(block_size=1024, vocab_size=50257, n_layers=24, n_heads=16, dim=1024, use_flash=args.flash),
            "d36": Config(block_size=1024, vocab_size=50257, n_layers=36, n_heads=20, dim=1280, use_flash=args.flash),
            "d48": Config(block_size=1024, vocab_size=50257, n_layers=48, n_heads=25, dim=1600, use_flash=args.flash),
        }[args.model]
        model = GPTModel(model_config)
    else:
        # load the GPT-2 model weights
        model = load_from_hf_weights(args.model)
    start_step = 0
    best_validation_loss: float | None = None

    checkpoint: dict | None = None
    if os.path.isfile(os.path.join(output_dir, "ckpt.pt")):
        print0("Loading existing checkpoint...")
        checkpoint = _load_checkpoint(output_dir, model, device)
        start_step = checkpoint["step"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]

    model.train()
    model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True  # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedDataLoader(args.train_dataset_path_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.val_dataset_path_bin:
        val_loader = DistributedDataLoader(args.val_dataset_path_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = configure_optimizers(model, weight_decay=args.weight_decay, learning_rate=args.learning_rate, betas=(0.9, 0.95), device_type=device, zero_stage=zero_stage)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.number_of_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.number_of_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0  # dummy value to print in inference-only mode

    for step in range(start_step, args.number_of_iterations + 1):
        t0 = time.time()
        last_step = step == args.number_of_iterations

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0 and step > 0 and (step % args.val_loss_every == 0 or last_step)) and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))

                if logger is not None:
                    logger.log_metrics({"ValLoss": val_loss}, step=step)

                if best_validation_loss is None or best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    _save_checkpoint(output_dir, model, optimizer, step, args, best_validation_loss)

        # once in a while perform model inference on the master process
        if (args.sample_every > 0 and (step % args.sample_every == 0 or last_step)) and master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            start_ids = [enc.eot_token]
            xg = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
            print0("---------------")
            print0(enc.decode(yg[0].tolist()))
            print0("---------------")

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # if we are trying to overfit a single batch, we reset the loader here
        if args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0  # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach()  # keep track of the mean loss
            # backward pass
            if not args.inference_only:
                loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1 - t0)
        print0(f"step {step+1:4d}/{args.number_of_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))
            if logger is not None:
                logger.log_metrics({"Loss": lossf}, step=step)

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.number_of_iterations - 20:
            timings.append(t1 - t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
    if logger is not None:
        logger.finalize()


if __name__ == "__main__":
    # experiments_dir = "experiments"
    # data_folder = r"C:\work\repos\ml-playground"
    # dataset_train_bin = os.path.join(data_folder, "train_bible.bin")
    # dataset_val_bin = os.path.join(data_folder, "val_bible.bin")
    # model = "d12"
    # experiment_name = "train_gptd12"

    # train_parameters = TrainParameters(experiment_name=experiment_name, model=model, output_dir=experiments_dir, batch_size=4, total_batch_size=2048, sequence_length=256, train_dataset_path_bin=dataset_train_bin, val_dataset_path_bin=dataset_val_bin, number_of_iterations=100000, sample_every=1000)
    # train(train_parameters)

    import os

    from os import listdir
    from os.path import isfile, join
    all_files_path = r"C:\work\repos\Autopilot.Samples\Dataset\WorkflowGeneration\Portable\all_data"
    onlyfiles = [f for f in listdir(all_files_path) if isfile(join(all_files_path, f))]
    
    from random import shuffle
    import shutil

    shuffle(onlyfiles)
    k_buckets = 5

    bucket_size = len(onlyfiles)//5
    indices = np.zeros(len(onlyfiles))
    onlyfiles = np.asarray(onlyfiles)

    test_folder = r"C:\work\repos\Autopilot.Samples\Dataset\WorkflowGeneration\Portable\all_data\test"
    train_folder = r"C:\work\repos\Autopilot.Samples\Dataset\WorkflowGeneration\Portable\all_data\train"

    def cleanup_folder(directory_path):
        try:
            files = os.listdir(directory_path)
            for file in files:
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("All files deleted successfully.")
        except OSError:
            print("Error occurred while deleting files.")


    

    for i in range(k_buckets):
        indices[:] = 0
        indices[i:i + bucket_size] = 1
        test_set = onlyfiles[indices == 1]
        train_set= onlyfiles[indices == 0]
        os.makedirs(test_folder,exist_ok=True)
        os.makedirs(train_folder,exist_ok=True)
        cleanup_folder(test_folder)
        cleanup_folder(train_folder)

        file_sets = [(test_folder, test_set),(train_folder, train_set)]

        for dest_folder, file_list in file_sets:
            for file in file_list:
                source_path = os.path.join(all_files_path, file)
                destination_path = os.path.join(dest_folder, file)
                shutil.copyfile(source_path, destination_path)

      
           


        


        


